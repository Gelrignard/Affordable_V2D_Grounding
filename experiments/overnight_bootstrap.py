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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--runtime', required=True)
    p.add_argument('--wait-queue', required=True)
    p.add_argument('--source-run', action='append', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--deadline-utc', default='2026-10-27T13:30:00Z')
    args = p.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline = parse_deadline(args.deadline_utc)
    with exclusive_lock(root / 'bootstrap.lock'):
        status_file = root / 'bootstrap_status.json'
        if status_file.exists():
            raise RuntimeError('Bootstrap has already been invoked; inspect existing state before any retry')
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
            baseline = evaluate(args, selected['policy'], 'parent_evaluation')
            baseline_scores = baseline['leaderboard']['scores']
            for name, old in selected['scores'].items():
                new = baseline_scores[name]
                if new is None or not math.isclose(new, old, rel_tol=1e-4, abs_tol=1e-4):
                    raise RuntimeError(f'Parent evaluation parity failed for {name}: {old} vs {new}')
            write_json(root / 'best_checkpoint.json', {**selected, 'source_commit': 'upstream_existing',
                       'overnight_evaluation': str(root / 'parent_evaluation/evaluation.json')})
            output = root / 'candidate1_screen'
            state, policy, metadata = checkpoint_pair(selected['state'])
            end, _ = transition_budget(selected['environment_steps'], 60_000_000)
            planned = build_config(json.loads(metadata['resolved_config_json']), state=state,
                                   output=output, start=selected['environment_steps'], end=end,
                                   phase='refine', checkpoint_interval=10_000_000)
            planned_path = root / 'candidate1_screen_planned_config.json'
            write_json(planned_path, planned)
            write_json(root / 'candidate1_screen_manifest.json', {
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
            run_guard(args, 'candidate1_screen', 'training', 1860, command)
            if read_json(output / 'resolved_config.json') != planned:
                raise RuntimeError('Executed training config differs from the prepared config')
            checkpoint = read_json(output / 'latest_checkpoint.json')
            if not checkpoint or checkpoint['environment_steps'] <= selected['environment_steps']:
                raise RuntimeError('Training did not produce a new checkpoint')
            status.update(state='evaluating_first_screen', checkpoint=checkpoint)
            write_json(status_file, status)
            result = evaluate(args, checkpoint['policy'], 'candidate1_screen_evaluation')
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
