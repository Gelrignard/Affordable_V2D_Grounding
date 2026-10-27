"""CPU-only input rebasing and immutable native-bootstrap plan for two episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from experiments.overnight_guard import parse_deadline, utc_now
from experiments.overnight_policy import build_fresh_config, OPTIMIZER_HORIZON
from training import RUNTIME_COMMIT


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def immutable_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_job(job, episode):
    job = Path(job).resolve()
    manifest = json.loads((job / 'job.json').read_text(encoding='utf-8'))
    if (manifest.get('schema'), manifest.get('tier'), manifest.get('episode_index'),
            manifest.get('variant'), manifest.get('runtime_commit')) != (
            'v2d_policy_job_v1', 1, episode, 'corrected', RUNTIME_COMMIT):
        raise ValueError('Unexpected task identity, variant or runtime')
    for name, expected in manifest['files'].items():
        path = (job / name).resolve()
        if not path.is_relative_to(job) or sha(path) != expected:
            raise ValueError(f'Invalid input file: {name}')
    return job, manifest


def prepare_episode(root, runtime, episode, job):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pxr import Usd
    from flash_chord.configuration import compose_config, resolved_dict, instantiate_typed
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.training.flash_sac.config import TrainingConfig
    from omegaconf import OmegaConf

    job, manifest = validate_job(job, episode)
    sequence = manifest['sequence_id']
    if not sequence or '/' in sequence or '\\' in sequence or sequence in ('.', '..'):
        raise ValueError('Invalid sequence component')
    prepared = root / 'prepared' / f'episode_{episode:06d}'
    prepared.mkdir(parents=True, exist_ok=False)
    motion_dir = prepared / 'input/retargeted' / f'sequence_id={sequence}' / 'robot_name=g1'
    motion_dir.mkdir(parents=True)
    rows = pq.read_table(job / 'motion.parquet').to_pylist()
    if len(rows) != 1 or rows[0]['sequence_id'] != sequence:
        raise ValueError('Job and motion identity differ')
    row = rows[0]
    row['object_mesh_paths'] = [str(job / 'assets/object.obj')]
    row['object_urdf_paths'] = [str(job / 'assets/object.urdf')]
    motion = motion_dir / 'motion.parquet'
    pq.write_table(pa.Table.from_pylist([row]), motion, compression='zstd')
    # The native loader discovers support from the sibling reconstructed_stage.
    support = job / 'assets/support.usda'
    stage = Usd.Stage.Open(str(support))
    if stage is None:
        raise ValueError('Cannot open support geometry')
    has_support = any(p.GetTypeName() in ('Cube', 'Cylinder') for p in stage.Traverse())
    if has_support:
        support_dir = prepared / 'input/reconstructed_stage'
        support_dir.mkdir()
        (support_dir / f'{sequence}_support.usda').symlink_to(support)
    source = runtime / 'robotic_grounding/flash_chord/src'
    urdf = source / 'flash_chord/assets/urdfs/g1/main_with_hand.urdf'
    overrides = [
        'experiment=g1_recon_body_flash_sac',
        f'embodiment.config.asset_sha256={sha(urdf)}',
        f'task.parquet={json.dumps(str(motion))}',
        f'scene.support={str(has_support).lower()}',
        'scene.world_count=2048', 'training.replay.batch_size=2048',
        'training.replay.capacity=4000000',
        f'training.optimization.learning_rate_schedule_updates={OPTIMIZER_HORIZON}',
        'logging.mode=offline', 'logging.log_interval=25',
        'logging.upload_checkpoints=false', 'logging.upload_training_state=false',
    ]
    cfg = resolved_dict(compose_config('train_flash_sac', overrides))
    bootstrap_output = root / f'episode_{episode:06d}_native'
    cfg = build_fresh_config(cfg, output=bootstrap_output, end=349999104,
                             checkpoint_interval=50000000)
    # Instantiate configuration objects only; no physics scene or GPU allocation.
    instantiate_typed(OmegaConf.create(cfg['env']), RLEnvConfig)
    instantiate_typed(OmegaConf.create(cfg['training']), TrainingConfig)
    config_path = prepared / 'native_config.json'
    immutable_json(config_path, cfg)
    input_manifest = {'prepared_utc': utc_now(), 'job': str(job),
                      'job_manifest_sha256': sha(job / 'job.json'),
                      'source_file_sha256': manifest['files'],
                      'source_motion_sha256': sha(job / 'motion.parquet'),
                      'runtime_motion': str(motion), 'runtime_motion_sha256': sha(motion),
                      'support': has_support, 'sequence_id': sequence,
                      'source_frames': manifest['source_frames'],
                      'source_fps': manifest['source_fps'],
                      'source_frame_playback': True, 'sim_control_fps': 50,
                      'estimated_force_guidance_used': False}
    input_path = prepared / 'input_manifest.json'
    immutable_json(input_path, input_manifest)
    return {'episode_index': episode, 'tier': 1,
            'candidate_id': f'episode_{episode:06d}', 'job': str(job),
            'sequence_id': sequence, 'source_frames': manifest['source_frames'],
            'bootstrap_output': str(bootstrap_output),
            'bootstrap_config': str(config_path), 'bootstrap_config_sha256': sha(config_path),
            'input_manifest': str(input_path), 'input_manifest_sha256': sha(input_path),
            'refinement_output': str(root / f'episode_{episode:06d}_precision_screen')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'source-commit', 'source-archive-sha256', 'deadline-utc', 'wait-queue'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--episode-job', action='append', required=True, help='EPISODE=ABSOLUTE_JOB')
    args = parser.parse_args()
    parse_deadline(args.deadline_utc)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('JAX_PLATFORMS') != 'cpu':
        parser.error('Prepare with CUDA_VISIBLE_DEVICES empty and JAX_PLATFORMS=cpu')
    root, runtime = Path(args.root).resolve(), Path(args.runtime).resolve()
    if (root / 'plan.json').exists() or (root / 'queue_status.json').exists():
        raise ValueError('Refuse to overwrite an existing plan or queue')
    if (SOURCE / 'SOURCE_COMMIT').read_text().strip() != args.source_commit:
        raise ValueError('Source deployment does not match supplied commit')
    actual = subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(runtime), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if actual != RUNTIME_COMMIT or dirty:
        raise ValueError('Native runtime must be pinned and clean')
    source = runtime / 'robotic_grounding/flash_chord/src'
    sys.path.insert(0, str(source))
    pairs = []
    for entry in args.episode_job:
        episode, job = entry.split('=', 1)
        pairs.append((int(episode), job))
    if [episode for episode, _ in pairs] != [16, 18]:
        raise ValueError('This authorized trial runs exactly episodes16 then18')
    source_files = ['experiments/prepare_two_episode_policy.py', 'experiments/two_episode_queue.py',
                    'experiments/overnight_policy.py', 'experiments/overnight_guard.py',
                    'experiments/evaluate_overnight_policy.py', 'training/overnight_refinement.py']
    plan = {'schema': 'v2d_two_episode_trial_v1', 'prepared_utc': utc_now(),
            'authorization': 'User requested the same method on two additional episodes, October27',
            'source_commit': args.source_commit, 'source_archive_sha256': args.source_archive_sha256,
            'source_file_sha256': {p: sha(SOURCE / p) for p in source_files},
            'runtime': str(runtime), 'runtime_commit': RUNTIME_COMMIT, 'root': str(root),
            'deadline_utc': args.deadline_utc, 'wait_queue': args.wait_queue,
            'total_budget_seconds': 23400, 'candidate_training_budget_seconds': 10800,
            'maximum_candidates': 2, 'bootstrap_transitions': 349999104,
            'bootstrap_wall_seconds': 9000, 'refinement_transitions': 60000000,
            'refinement_wall_seconds': 1800, 'evaluation_worlds': 16,
            'automatic_promotion': False, 'leave_instance_powered_on': True,
            'episodes': [prepare_episode(root, runtime, episode, job) for episode, job in pairs]}
    immutable_json(root / 'plan.json', plan)
    print(json.dumps({'plan': str(root / 'plan.json'), 'sha256': sha(root / 'plan.json'),
                      'episodes': [e['episode_index'] for e in plan['episodes']]}))


if __name__ == '__main__':
    main()
