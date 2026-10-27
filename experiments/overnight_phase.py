"""Run one explicitly selected overnight training phase, evaluate, then stop."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from experiments.overnight_bootstrap import file_hash
from experiments.overnight_guard import (
    exclusive_lock, parse_deadline, pid_live, read_json, remaining_budget, utc_now, write_json,
)
from experiments.overnight_policy import build_config, checkpoint_pair, transition_budget


def component(value):
    if not value or not value.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Candidate and phase names must be single alphanumeric path components')
    return value


def immutable_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parent_scores(report, policy, steps):
    expected = {'checkpoint_sha256': file_hash(policy), 'environment_steps': steps,
                'world_count': 16, 'start_frame': 0, 'reset_mode': 'explicit', 'settling_steps': 0}
    if any(report.get(key) != value for key, value in expected.items()) or report.get('non_finite_worlds'):
        raise ValueError('Parent evaluation must match the checkpoint and finite explicit frame-zero cohort')
    contract = report.get('evaluation_contract', {})
    if contract.get('voc_scale', report.get('probe_assistance')) != 0:
        raise ValueError('Parent evaluation must be unassisted')
    if contract and (contract.get('reset_voc_scale') != 0 or contract.get('within_attempt_resets') != 0):
        raise ValueError('Parent rollout must use zero reset assistance and no within-attempt resets')
    scores = report['leaderboard']['scores']
    if any(not isinstance(scores.get(key), (float, int)) or not math.isfinite(scores[key])
           for key in ('add_auc', 'mppe_cm', 'spider_sr', 'maniptrans_sr')):
        raise ValueError('Parent evaluation scores must be finite')
    return scores


def guarded(args, name, kind, seconds, command):
    root = Path(args.root)
    subprocess.run([
        sys.executable, str(SOURCE / 'experiments/overnight_guard.py'),
        '--ledger', str(root / 'ledger.json'), '--name', name, '--candidate', args.candidate,
        '--kind', kind, '--max-seconds', str(seconds), '--deadline-utc', args.deadline_utc,
        '--wait-queue', args.wait_queue, '--wait-seconds', '60', '--cwd', str(SOURCE),
        '--output', str(root / 'supervision' / name), '--', *command,
    ], check=True)


def run(args):
    component(args.name)
    component(args.candidate)
    if not 60 <= args.max_training_seconds <= 7200:
        raise ValueError('Training duration must be between 60 and 7200 seconds')
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    args.root = str(root)
    deadline = parse_deadline(args.deadline_utc)
    with exclusive_lock(root / 'phase.lock'):
        bootstrap = read_json(root / 'bootstrap_status.json', {})
        if bootstrap.get('state') not in ('awaiting_agent_diagnosis', 'needs_attention') or pid_live(bootstrap.get('pid')):
            raise RuntimeError('Bootstrap must be stopped before a continuation phase')
        ledger = read_json(root / 'ledger.json', {'jobs': []})
        seconds = int(min(args.max_training_seconds,
                          remaining_budget(ledger, args.candidate, args.max_training_seconds),
                          deadline - time.time() - 600))
        if seconds < 60 or remaining_budget(ledger, args.candidate, seconds + 600, kind='evaluation') < seconds + 600:
            raise RuntimeError('Insufficient budget for training and reserved evaluation')
        output, evaluation = root / args.name, root / (args.name + '_evaluation')
        status_path = root / (args.name + '_status.json')
        plan_path, manifest_path = root / (args.name + '_planned_config.json'), root / (args.name + '_manifest.json')
        reserved = [output, evaluation, status_path, plan_path, manifest_path,
                    root / (args.name + '_comparison.json'), root / 'supervision' / args.name,
                    root / 'supervision' / evaluation.name]
        if any(path.exists() for path in reserved):
            raise RuntimeError('Phase name already has artifacts; choose a new name and preserve them')
        state, policy, metadata = checkpoint_pair(args.parent_state)
        start = int(metadata['environment_steps'])
        baseline = read_json(args.parent_evaluation)
        scores = parent_scores(baseline or {}, policy, start)
        end, rounded = transition_budget(start, args.additional_transitions)
        planned = build_config(json.loads(metadata['resolved_config_json']), state=state, output=output,
                               start=start, end=end, phase=args.phase, checkpoint_interval=10_000_000)
        immutable_json(plan_path, planned)
        source_paths = ('experiments/overnight_phase.py', 'experiments/overnight_policy.py',
                        'experiments/overnight_guard.py', 'experiments/evaluate_overnight_policy.py',
                        'training/overnight_refinement.py')
        immutable_json(manifest_path, {
            'source_commit': args.source_commit, 'source_directory': str(SOURCE), 'prepared_utc': utc_now(),
            'source_file_sha256': {path: file_hash(SOURCE / path) for path in source_paths},
            'parent_state': str(state), 'parent_state_sha256': file_hash(state),
            'parent_policy': str(policy), 'parent_policy_sha256': file_hash(policy),
            'parent_evaluation': str(Path(args.parent_evaluation).resolve()),
            'parent_evaluation_sha256': file_hash(args.parent_evaluation),
            'effective_config': str(plan_path), 'effective_config_sha256': file_hash(plan_path),
            'motion_parquet': planned['task']['parquet'], 'motion_parquet_sha256': file_hash(planned['task']['parquet']),
            'candidate': args.candidate, 'phase': args.phase, 'additional_transitions': rounded,
            'max_training_seconds': seconds, 'evaluation_seconds_cap': 600, 'deadline_utc': args.deadline_utc,
            'fresh_replay': True, 'automatic_promotion': False,
        })
        status = {'state': 'training', 'pid': os.getpid(), 'started_utc': utc_now(),
                  'source_commit': args.source_commit, 'candidate': args.candidate, 'phase': args.phase,
                  'parent_state': str(state), 'output': str(output), 'arguments': vars(args)}
        write_json(status_path, status)
        try:
            guarded(args, args.name, 'training', min(seconds + 30, 7200), [
                sys.executable, str(SOURCE / 'experiments/overnight_policy.py'), 'train',
                '--runtime', args.runtime, '--checkpoint', str(state), '--output', str(output),
                '--phase', args.phase, '--additional-transitions', str(rounded),
                '--max-seconds', str(seconds), '--deadline-utc', args.deadline_utc,
            ])
            if read_json(output / 'resolved_config.json') != planned:
                raise RuntimeError('Executed training configuration differs from the immutable plan')
            latest = read_json(output / 'latest_checkpoint.json')
            if not latest or int(latest['environment_steps']) <= start:
                raise RuntimeError('Phase produced no checkpoint with additional transitions')
            _, latest_policy, latest_metadata = checkpoint_pair(latest['state'])
            if str(latest_policy) != latest['policy'] or int(latest_metadata['environment_steps']) != latest['environment_steps']:
                raise RuntimeError('Latest checkpoint marker and policy/state pair disagree')
            status.update(state='evaluating', checkpoint=latest)
            write_json(status_path, status)
            guarded(args, evaluation.name, 'evaluation', 600, [
                sys.executable, str(SOURCE / 'experiments/evaluate_overnight_policy.py'),
                '--runtime', args.runtime, '--checkpoint', str(latest_policy), '--output', str(evaluation),
                '--worlds', '16', '--tier', '1', '--episode', '5', '--variant', args.name,
            ])
            result = read_json(evaluation / 'evaluation.json')
            comparison = {'parent': scores, 'candidate': result['leaderboard']['scores'],
                          'parent_evaluation': str(Path(args.parent_evaluation).resolve()),
                          'candidate_evaluation': str(evaluation / 'evaluation.json'),
                          'parent_phase_metrics': baseline.get('phase_metrics'), 'phase_metrics': result['phase_metrics'],
                          'needs_agent_diagnosis': True, 'automatic_promotion': False, 'source_commit': args.source_commit}
            immutable_json(root / (args.name + '_comparison.json'), comparison)
            status.update(state='awaiting_agent_diagnosis', completed_utc=utc_now(),
                          comparison=str(root / (args.name + '_comparison.json')))
            write_json(status_path, status)
        except BaseException as exc:
            status.update(state='needs_attention', error=f'{type(exc).__name__}: {exc}', updated_utc=utc_now())
            write_json(status_path, status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'source-commit', 'parent-state', 'parent-evaluation',
                 'candidate', 'name', 'deadline-utc', 'wait-queue'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--phase', choices=('refine', 'consolidate'), required=True)
    parser.add_argument('--additional-transitions', type=int, required=True)
    parser.add_argument('--max-training-seconds', type=int, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
