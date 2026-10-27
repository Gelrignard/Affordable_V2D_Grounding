"""Wait for the existing queue, then run one bounded refinement screen.

The agent reads the resulting full-rollout diagnostics before deciding the next
training phase. This detached starter does not invent further experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from experiments.overnight_guard import (
    exclusive_lock, external_queue_ready, parse_deadline, read_json, utc_now, write_json,
)
from experiments.overnight_policy import build_config, checkpoint_pair, transition_budget


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def eligible_sources(run_directories):
    candidates, rejected = [], []
    for run in map(Path, run_directories):
        for report in sorted((run / 'evaluations').glob('*/evaluation.json')):
            try:
                data = read_json(report)
                if data.get('non_finite_worlds') or data.get('world_count') != 16:
                    raise ValueError('Requires finite 16-world evaluation')
                if data.get('start_frame') != 0 or data.get('reset_mode') != 'explicit':
                    raise ValueError('Requires explicit frame-zero evaluation')
                if data.get('probe_assistance') != 0 or data.get('settling_steps') != 0:
                    raise ValueError('Requires unassisted evaluation without settling')
                policy = Path(data['checkpoint'])
                state, policy, metadata = checkpoint_pair(policy)
                if int(metadata['environment_steps']) != int(data['environment_steps']):
                    raise ValueError('Evaluation and checkpoint steps differ')
                scores = data['leaderboard']['scores']
                if not all(math.isfinite(scores[k]) for k in ('add_auc', 'mppe_cm', 'spider_sr', 'maniptrans_sr')):
                    raise ValueError('Evaluation scores are not finite')
                candidates.append({'state': str(state), 'policy': str(policy),
                                   'evaluation': str(report), 'scores': scores,
                                   'environment_steps': int(metadata['environment_steps'])})
            except (ValueError, KeyError, OSError, TypeError) as exc:
                rejected.append({'evaluation': str(report), 'reason': str(exc)})
    # Use the best available evaluated AUC, not the last saved policy. A complete
    # policy/state pair is required so critic/optimizer restoration is possible.
    candidates.sort(key=lambda c: (c['scores']['add_auc'], -c['scores']['mppe_cm']), reverse=True)
    if not candidates:
        raise ValueError(f'No evaluated compact state is available: {rejected}')
    return candidates, rejected


def run_guard(args, name, kind, seconds, command):
    root = Path(args.root)
    invocation = [sys.executable, str(SOURCE / 'experiments/overnight_guard.py'),
                  '--ledger', str(root / 'ledger.json'), '--name', name,
                  '--candidate', 'candidate1', '--kind', kind, '--max-seconds', str(seconds),
                  '--deadline-utc', args.deadline_utc, '--wait-queue', args.wait_queue,
                  '--wait-seconds', '60', '--cwd', str(SOURCE),
                  '--output', str(root / 'supervision' / name), '--', *command]
    subprocess.run(invocation, check=True)


def evaluate(args, checkpoint, name):
    output = Path(args.root) / name
    command = [sys.executable, str(SOURCE / 'experiments/evaluate_overnight_policy.py'),
               '--runtime', args.runtime, '--checkpoint', str(checkpoint),
               '--output', str(output), '--worlds', '16', '--tier', '1', '--episode', '5',
               '--variant', name]
    run_guard(args, name, 'evaluation', 600, command)
    return read_json(output / 'evaluation.json')


def verify_parent_repeat(expected, actual):
    """Verify the evaluation contract without asserting bit-exact GPU physics."""
    identity = ('checkpoint_sha256', 'world_count', 'step_count', 'start_frame',
                'reset_mode', 'settling_steps', 'sim_control_fps', 'sim_physics_fps', 'parquet')
    for name in identity:
        if name not in expected or expected[name] != actual.get(name):
            raise RuntimeError(f'Parent evaluation contract differs: {name}')
    if actual.get('non_finite_worlds'):
        raise RuntimeError('Parent evaluation has non-finite worlds')
    tolerances = {'add_auc': .01, 'mppe_cm': .5, 'spider_sr': 0., 'maniptrans_sr': 0.}
    deltas = {}
    for name, tolerance in tolerances.items():
        old, new = expected['leaderboard']['scores'][name], actual['leaderboard']['scores'][name]
        if not isinstance(new, (float, int)) or not math.isfinite(new):
            raise RuntimeError(f'Parent repeat score is invalid: {name}')
        deltas[name] = new - old
        if abs(deltas[name]) > tolerance:
            raise RuntimeError(f'Parent repeat drift exceeds inspection bound for {name}: {old} vs {new}')
    return {'identity_fields_verified': list(identity), 'score_differences': deltas,
            'absolute_inspection_bounds': tolerances,
            'comparison_baseline': 'new_evaluator_parent_rollout',
            'note': 'Small repeat variation accepted; this is not an improvement or a deterministic replay claim.'}


def _tensor_fingerprints(path):
    """Read the pinned safetensors envelope and hash payloads using only stdlib."""
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise RuntimeError('Truncated checkpoint header')
        size = struct.unpack('<Q', raw)[0]
        if not 0 < size <= 32 * 1024 * 1024:
            raise RuntimeError('Invalid checkpoint header size')
        header = json.loads(stream.read(size))
        metadata = header['__metadata__']
        if metadata.get('format') != 'learner_state' or metadata.get('version') != '1':
            raise RuntimeError('Unsupported checkpoint envelope')
        manifest = json.loads(metadata['manifest'])
        keys = [entry['key'] for entry in manifest]
        if not keys or keys != [f'state.{i:04d}' for i in range(len(keys))]:
            raise RuntimeError('Expected the pinned contiguous native leaf manifest')
        if set(header) - {'__metadata__'} != set(keys):
            raise RuntimeError('Checkpoint manifest does not cover all tensors')
        data_start, payload_size = 8 + size, path.stat().st_size - 8 - size
        result = {}
        for entry in manifest:
            key = entry['key']
            spec = header[key]
            begin, end = spec['data_offsets']
            if not 0 <= begin <= end <= payload_size:
                raise RuntimeError(f'Invalid tensor payload bounds: {key}')
            expected_shape = entry['shape'] or [1]
            if spec['shape'] != expected_shape:
                raise RuntimeError(f'Tensor header/manifest shape differs: {key}')
            stream.seek(data_start + begin)
            digest = hashlib.sha256()
            remaining = end - begin
            scalar = b''
            while remaining:
                block = stream.read(min(remaining, 1024 * 1024))
                if not block:
                    raise RuntimeError('Truncated tensor payload')
                digest.update(block)
                if end - begin <= 16:
                    scalar += block
                remaining -= len(block)
            result[key] = {'sha256': digest.hexdigest(), 'dtype': spec['dtype'],
                           'shape': entry['shape'], 'scalar': scalar}
    return manifest, result


def _integer_leaf(tensor):
    formats = {'I32': '<i', 'I64': '<q', 'U32': '<I', 'U64': '<Q'}
    if tensor['shape'] != [] or tensor['dtype'] not in formats:
        raise RuntimeError('Expected a scalar integer checkpoint clock')
    return struct.unpack(formats[tensor['dtype']], tensor['scalar'])[0]


def verify_unlearned_startup(parent_checkpoint, output):
    """Permit discarding only a proven fresh-replay warm-up with no learning.

This is deliberately tied to the pinned TrainingCheckpoint layout, not a generic
checkpoint recovery mechanism. Every actor/critic/target/temperature leaf,
including optimizer moments and scheduler clocks, must remain byte-identical.
"""
    from training import RUNTIME_COMMIT

    output = Path(output)
    latest = read_json(output / 'latest_checkpoint.json')
    provenance = read_json(output / 'provenance.json')
    parity = read_json(output / 'warmstart_parity.json')
    config = read_json(output / 'resolved_config.json')
    if not all((latest, provenance, parity, config)):
        raise RuntimeError('Failed startup lacks checkpoint/provenance/parity/config evidence')
    parent_state, parent_policy, before = checkpoint_pair(parent_checkpoint)
    failed_state, failed_policy, after = checkpoint_pair(latest['state'])
    if Path(latest['policy']).resolve() != failed_policy:
        raise RuntimeError('Failed startup latest policy/state pair differs')
    if provenance.get('runtime_commit') != RUNTIME_COMMIT or provenance.get('replay_is_fresh') is not True:
        raise RuntimeError('Failed startup must use pinned runtime and fresh replay')
    if (Path(provenance['source_state']).resolve() != parent_state
            or provenance.get('source_state_sha256') != file_hash(parent_state)
            or provenance.get('source_policy_sha256') != file_hash(parent_policy)):
        raise RuntimeError('Failed startup parent provenance differs')
    if (parity.get('parameter_exact_match') is not True
            or parity.get('deterministic_action_max_abs_error') != 0
            or parity.get('fresh_replay_size') != 0):
        raise RuntimeError('Failed startup lacks exact parent restoration and empty replay proof')
    for name in ('policy_schema_json', 'critic_schema_json', 'global_update_step'):
        if name not in before or before[name] != after.get(name):
            raise RuntimeError(f'Failed startup changed or omitted {name}')
    start, stopped = int(before['environment_steps']), int(after['environment_steps'])
    added = stopped - start
    minimum = int(config['training']['replay']['minimum_size'])
    if not 0 <= added < minimum:
        raise RuntimeError('Failed startup is outside the fresh-replay warm-up window')
    if parity.get('environment_steps') != start or parity.get('global_update_step') != int(before['global_update_step']):
        raise RuntimeError('Warm-start clocks do not match the parent')
    if int(latest['environment_steps']) != stopped:
        raise RuntimeError('Failed checkpoint and latest marker disagree')
    before_manifest, before_tensors = _tensor_fingerprints(parent_state)
    after_manifest, after_tensors = _tensor_fingerprints(failed_state)
    if before_manifest != after_manifest or len(before_manifest) <= 14:
        raise RuntimeError('Compact checkpoint layout differs or is incomplete')
    keys = [entry['key'] for entry in before_manifest]
    if any((before_tensors[key]['dtype'], before_tensors[key]['shape']) !=
           (after_tensors[key]['dtype'], after_tensors[key]['shape']) for key in keys):
        raise RuntimeError('Compact checkpoint tensor schema changed')
    # Native TrainingCheckpoint fields after temperature: normalizer(5),
    # exploration(3), loss_scale(2), key, update_credit, env_steps, global_update.
    tail = keys[-14:]
    expected_tail = [('F32', 1), *[('F32', 0)] * 4, ('F32', 2),
                     ('I32', 0), ('I32', 0), ('F32', 0), ('I32', 0),
                     ('U32', 1), ('F32', 0), ('I32', 0), ('I32', 0)]
    for key, (dtype, dimensions) in zip(tail, expected_tail, strict=True):
        tensor = before_tensors[key]
        if tensor['dtype'] != dtype or len(tensor['shape']) != dimensions:
            raise RuntimeError(f'Unrecognized pinned compact tail layout at {key}')
    if before_tensors[tail[0]]['shape'][0] != before_tensors[tail[5]]['shape'][0] or before_tensors[tail[10]]['shape'] != [2]:
        raise RuntimeError('Unrecognized normalizer/exploration/PRNG shapes')
    allowed = set(tail[:8] + [tail[10], tail[11], tail[12]])
    protected = [key for key in keys if key not in allowed]
    for key in protected:
        if before_tensors[key] != after_tensors[key]:
            raise RuntimeError(f'Failed startup changed protected model/optimizer/clock tensor {key}')
    if (_integer_leaf(before_tensors[tail[-1]]) != int(before['global_update_step'])
            or _integer_leaf(before_tensors[tail[-2]]) != start
            or _integer_leaf(after_tensors[tail[-2]]) != stopped):
        raise RuntimeError('Checkpoint scalar clocks disagree with metadata')
    policy_manifest, policy_tensors = _tensor_fingerprints(parent_policy)
    failed_policy_manifest, failed_policy_tensors = _tensor_fingerprints(failed_policy)
    if policy_manifest != failed_policy_manifest or policy_tensors != failed_policy_tensors:
        raise RuntimeError('Failed startup changed deployable actor tensors')
    return {'verified_zero_learning': True, 'parent_state': str(parent_state),
            'failed_state': str(failed_state), 'failed_state_sha256': file_hash(failed_state),
            'parent_state_sha256': file_hash(parent_state), 'additional_transitions_discarded': added,
            'fresh_replay_minimum_size': minimum, 'global_update_step': int(before['global_update_step']),
            'protected_state_tensor_count': len(protected), 'policy_tensor_count': len(policy_manifest),
            'protected_tensor_sha256': {key: before_tensors[key]['sha256'] for key in protected},
            'changed_discarded_warmup_leaves': [key for key in keys if key in allowed and before_tensors[key] != after_tensors[key]],
            'discarded_state': 'Fresh replay, reward normalizer updates, exploration/RNG, update credit and warm-up environment steps; retry restores the original parent.',
            'failed_artifacts_preserved': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--runtime', required=True)
    p.add_argument('--wait-queue', required=True)
    p.add_argument('--source-run', action='append', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--deadline-utc', default='2026-10-27T13:30:00Z')
    p.add_argument('--resume-parent-evaluation', action='store_true',
                   help='Resume a stopped initial parent check using its completed saved rollout')
    p.add_argument('--screen-name', default='candidate1_screen',
                   help='Unique output name for the initial screen or a failed-startup retry')
    args = p.parse_args()
    if not args.screen_name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Screen name must be a single alphanumeric path component')
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline = parse_deadline(args.deadline_utc)
    with exclusive_lock(root / 'bootstrap.lock'):
        status_file = root / 'bootstrap_status.json'
        previous = read_json(status_file)
        if previous and not args.resume_parent_evaluation:
            raise RuntimeError('Bootstrap has already been invoked; inspect existing state before any retry')
        if args.resume_parent_evaluation:
            if not previous or previous.get('state') != 'needs_attention':
                raise RuntimeError('Parent-check resume requires a stopped bootstrap')
            ledger = read_json(root / 'ledger.json', {'jobs': []})
            if any(j.get('state') == 'running' for j in ledger['jobs']):
                raise RuntimeError('Parent-check resume cannot duplicate running work')
            startup_verifications = []
            for job in ledger['jobs']:
                if job.get('kind') != 'training':
                    continue
                command = job['command']
                output = Path(command[command.index('--output') + 1])
                if job.get('state') == 'complete':
                    raise RuntimeError('Inspect existing trained checkpoints instead of restarting from parent')
                if (output / 'latest_checkpoint.json').exists():
                    stopped_failed_child = (job.get('state') == 'stopped'
                                            and read_json(output / 'status.json', {}).get('state') == 'failed')
                    if job.get('state') != 'failed' and not stopped_failed_child:
                        raise RuntimeError('Only a failed startup can discard verified unlearned warm-up')
                    parent = command[command.index('--checkpoint') + 1]
                    startup_verifications.append(verify_unlearned_startup(parent, output))
            if (root / args.screen_name).exists():
                raise RuntimeError('Retry needs a new output directory; preserve the failed attempt')
            write_json(root / f'bootstrap_previous_{int(time.time())}.json', previous)
            if startup_verifications:
                write_json(root / f'{args.screen_name}_failed_startup_verification.json',
                           {'attempts': startup_verifications, 'verified_utc': utc_now()})
        status = {'state': 'waiting_for_existing_queue', 'started_utc': utc_now(),
                  'pid': os.getpid(), 'source_commit': args.source_commit,
                  'source': str(SOURCE), 'arguments': vars(args)}
        write_json(status_file, status)
        try:
            while True:
                ready, reason = external_queue_ready(args.wait_queue)
                status.update(wait_reason=reason, updated_utc=utc_now())
                write_json(status_file, status)
                if ready:
                    if deadline - time.time() < 45 * 60:
                        raise RuntimeError('Insufficient remaining time for first screen; no GPU work started')
                    break
                if deadline - time.time() < 45 * 60:
                    raise RuntimeError('Insufficient remaining time for first screen; no GPU work started')
                time.sleep(30)
            candidates, rejected = eligible_sources(args.source_run)
            selected = candidates[0]
            write_json(root / 'starting_checkpoint.json', {'selected': selected,
                       'eligible': candidates, 'rejected': rejected, 'selected_utc': utc_now(),
                       'selection_rule': 'maximum available evaluated ADD-AUC; lower MPPE breaks ties'})
            status.update(state='evaluating_parent', parent=selected)
            write_json(status_file, status)
            baseline = (read_json(root / 'parent_evaluation/evaluation.json')
                        if args.resume_parent_evaluation else evaluate(args, selected['policy'], 'parent_evaluation'))
            if not baseline:
                raise RuntimeError('No completed parent evaluation is available')
            baseline_scores = baseline['leaderboard']['scores']
            repeat = verify_parent_repeat(read_json(selected['evaluation']), baseline)
            write_json(root / 'parent_repeat_verification.json', repeat)
            write_json(root / 'best_checkpoint.json', {**selected, 'scores': baseline_scores,
                       'historical_scores': selected['scores'], 'source_commit': 'upstream_existing',
                       'overnight_evaluation': str(root / 'parent_evaluation/evaluation.json')})
            output = root / args.screen_name
            state, policy, metadata = checkpoint_pair(selected['state'])
            end, _ = transition_budget(selected['environment_steps'], 60_000_000)
            planned = build_config(json.loads(metadata['resolved_config_json']), state=state,
                                   output=output, start=selected['environment_steps'], end=end,
                                   phase='refine', checkpoint_interval=10_000_000)
            planned_path = root / f'{args.screen_name}_planned_config.json'
            write_json(planned_path, planned)
            write_json(root / f'{args.screen_name}_manifest.json', {
                'source_commit': args.source_commit, 'source_directory': str(SOURCE),
                'prepared_utc': utc_now(), 'effective_config': str(planned_path),
                'effective_config_sha256': file_hash(planned_path),
                'parent_state': str(state), 'parent_state_sha256': file_hash(state),
                'parent_policy': str(policy), 'parent_policy_sha256': file_hash(policy),
                'motion_parquet': planned['task']['parquet'],
                'motion_parquet_sha256': file_hash(planned['task']['parquet']),
                'fresh_replay': True, 'training_voc_scale': 0,
                'recovery_reset_voc_scale': 1, 'evaluation_reset_voc_scale': 0,
            })
            status.update(state='training_first_screen', output=str(output))
            write_json(status_file, status)
            command = [sys.executable, str(SOURCE / 'experiments/overnight_policy.py'), 'train',
                       '--runtime', args.runtime, '--checkpoint', selected['state'],
                       '--output', str(output), '--phase', 'refine',
                       '--additional-transitions', '60000000', '--max-seconds', '1800',
                       '--deadline-utc', args.deadline_utc]
            run_guard(args, args.screen_name, 'training', 1860, command)
            if read_json(output / 'resolved_config.json') != planned:
                raise RuntimeError('Executed training config differs from the prepared config')
            checkpoint = read_json(output / 'latest_checkpoint.json')
            if not checkpoint or checkpoint['environment_steps'] <= selected['environment_steps']:
                raise RuntimeError('Training did not produce a new checkpoint')
            status.update(state='evaluating_first_screen', checkpoint=checkpoint)
            write_json(status_file, status)
            result = evaluate(args, checkpoint['policy'], args.screen_name + '_evaluation')
            comparison = {'parent': baseline_scores, 'candidate': result['leaderboard']['scores'],
                          'phase_metrics': result['phase_metrics'],
                          'needs_agent_diagnosis': True, 'source_commit': args.source_commit}
            write_json(root / 'first_screen_comparison.json', comparison)
            status.update(state='awaiting_agent_diagnosis', completed_utc=utc_now(),
                          comparison=str(root / 'first_screen_comparison.json'))
            write_json(status_file, status)
        except BaseException as exc:
            status.update(state='needs_attention', error=f'{type(exc).__name__}: {exc}', updated_utc=utc_now())
            write_json(status_file, status)
            raise


if __name__ == '__main__':
    main()
