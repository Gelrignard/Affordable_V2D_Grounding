"""Standard-library cloud supervisor: serialized jobs and durable wall budgets.

This module never imports CUDA. It never stops jobs it did not start. A blocked
queue is not a GPU run and is not charged to the additional-GPU-time budget.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

UTC = dt.timezone.utc
MAXIMUM_GPU_SECONDS = 6 * 3600
MAXIMUM_CANDIDATE_TRAINING_SECONDS = 2 * 3600
MAXIMUM_CANDIDATES = 3


def utc_now():
    return dt.datetime.now(UTC).isoformat()


def parse_deadline(value):
    value = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Deadline must have an explicit UTC offset')
    return value.timestamp()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def make_budget_policy(total_budget_seconds=MAXIMUM_GPU_SECONDS,
                       candidate_training_budget_seconds=MAXIMUM_CANDIDATE_TRAINING_SECONDS,
                       maximum_candidates=MAXIMUM_CANDIDATES):
    """Canonical, finite budget settings; defaults preserve the original scope."""
    for name, value in [('total_budget_seconds', total_budget_seconds),
                        ('candidate_training_budget_seconds', candidate_training_budget_seconds)]:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be a positive finite number')
    if isinstance(maximum_candidates, bool) or not isinstance(maximum_candidates, int) or maximum_candidates < 1:
        raise ValueError('maximum_candidates must be a positive integer')
    return {'schema': 'v2d_gpu_budget_policy_v1',
            'total_budget_seconds': float(total_budget_seconds),
            'candidate_training_budget_seconds': float(candidate_training_budget_seconds),
            'maximum_candidates': maximum_candidates}


def ledger_budget_policy(ledger):
    """Missing policy always means legacy limits, never a new expanded scope."""
    if 'budget_policy' not in ledger:
        return make_budget_policy()
    policy = ledger['budget_policy']
    if not isinstance(policy, dict) or set(policy) != set(make_budget_policy()):
        raise RuntimeError('Ledger budget_policy is malformed; fail closed')
    try:
        canonical = make_budget_policy(policy['total_budget_seconds'],
                                       policy['candidate_training_budget_seconds'],
                                       policy['maximum_candidates'])
    except ValueError as exc:
        raise RuntimeError('Ledger budget_policy is invalid; fail closed') from exc
    if policy != canonical:
        raise RuntimeError('Ledger budget_policy schema is unsupported; fail closed')
    return canonical


def ensure_budget_policy(ledger, requested_policy, *, ledger_exists):
    """Pin new ledgers; existing policies (including legacy) must match exactly.

    The caller holds the ledger lock and persists this before starting or waiting
    for a job. A policy-less existing ledger cannot acquire increased limits.
    """
    requested = ledger_budget_policy({'budget_policy': requested_policy})
    if ledger_exists or 'budget_policy' in ledger:
        existing = ledger_budget_policy(ledger)
        if requested != existing:
            raise RuntimeError('Requested budget policy differs from immutable ledger policy; '
                               'use a new ledger for a separately authorized scope')
    ledger['budget_policy'] = requested
    return requested


def remaining_budget(ledger, candidate, requested_seconds, kind='training'):
    jobs = ledger.get('jobs', [])
    if any(j.get('state') == 'running' for j in jobs):
        raise RuntimeError('Ledger contains a running job; reconcile its PID before launching')
    policy = ledger_budget_policy(ledger)
    used = sum(float(j.get('elapsed_seconds', 0)) for j in jobs)
    candidates = {j['candidate'] for j in jobs if j.get('kind') == 'training'}
    if kind == 'training' and candidate not in candidates and len(candidates) >= policy['maximum_candidates']:
        raise RuntimeError(f"Candidate implementation limit ({policy['maximum_candidates']}) reached")
    available = policy['total_budget_seconds'] - used
    if kind == 'training':
        trained = sum(float(j.get('elapsed_seconds', 0)) for j in jobs
                      if j.get('candidate') == candidate and j.get('kind') == 'training')
        available = min(available, policy['candidate_training_budget_seconds'] - trained)
    return max(0.0, min(float(requested_seconds), available))


def pid_live(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        status = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return status[0] != 'Z'
    except FileNotFoundError:
        return False


def gpu_pids():
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=15, check=True,
    )
    return [int(s.strip()) for s in result.stdout.splitlines() if s.strip().isdigit()]


def external_queue_ready(queue):
    status = read_json(Path(queue) / 'queue_status.json')
    if not status:
        return False, 'Existing queue status missing; fail closed'
    if status.get('state') not in {'complete', 'complete_with_evaluation_errors', 'failed',
                                  'stopped_at_wall_guard'}:
        return False, f"Existing queue state={status.get('state')}"
    if pid_live(status.get('pid')):
        return False, 'Existing queue supervisor still alive'
    pids = gpu_pids()
    if pids:
        return False, f'Other GPU processes still present: {pids}'
    return True, 'Existing queue terminal and GPU idle'


@contextlib.contextmanager
def exclusive_lock(path):
    import fcntl  # Cloud/Linux only; budget helpers remain testable on Windows.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another overnight worker owns the GPU lock') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def stop_child(proc, grace_seconds):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=max(0.1, grace_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run(args):
    if not args.command:
        raise ValueError('Pass a command after --')
    command = args.command[1:] if args.command[0] == '--' else args.command
    deadline = parse_deadline(args.deadline_utc)
    ledger_path = Path(args.ledger).resolve()
    with exclusive_lock(ledger_path.with_suffix('.lock')):
        ledger_exists = ledger_path.exists()
        ledger = read_json(ledger_path, {'created_utc': utc_now(), 'jobs': []})
        requested_policy = make_budget_policy(
            getattr(args, 'total_budget_seconds', MAXIMUM_GPU_SECONDS),
            getattr(args, 'candidate_training_budget_seconds', MAXIMUM_CANDIDATE_TRAINING_SECONDS),
            getattr(args, 'maximum_candidates', MAXIMUM_CANDIDATES))
        ensure_budget_policy(ledger, requested_policy, ledger_exists=ledger_exists)
        allowed = remaining_budget(ledger, args.candidate, args.max_seconds, args.kind)
        if allowed < 60 or deadline - time.time() < 60:
            raise RuntimeError('Insufficient remaining wall/calendar budget')
        write_json(ledger_path, ledger)
        waited = time.monotonic()
        wait_status = ledger_path.with_name('wait_status.json')
        while True:
            ready, reason = external_queue_ready(args.wait_queue)
            write_json(wait_status, {'state': 'ready' if ready else 'waiting',
                                    'reason': reason, 'pid': os.getpid(), 'time_utc': utc_now()})
            if ready:
                # Guard against a queue's just-exited training process preceding evaluation.
                time.sleep(2)
                ready, reason = external_queue_ready(args.wait_queue)
                if ready:
                    break
            if time.monotonic() - waited >= args.wait_seconds:
                raise RuntimeError(reason)
            if deadline - time.time() < 120:
                raise RuntimeError('Deadline approached while waiting for other work')
            time.sleep(min(30, max(1, args.wait_seconds - (time.monotonic() - waited))))
        allowed = min(allowed, deadline - time.time())
        if allowed < 60:
            raise RuntimeError('Insufficient wall/calendar budget after waiting')
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        job = {'name': args.name, 'candidate': args.candidate, 'kind': args.kind,
               'state': 'running', 'started_utc': utc_now(), 'elapsed_seconds': 0.0,
               'maximum_seconds': allowed, 'deadline_utc': args.deadline_utc,
               'command': command, 'output': str(output), 'supervisor_pid': os.getpid()}
        ledger['jobs'].append(job)
        write_json(ledger_path, ledger)
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', V2D_OVERNIGHT_GUARD='1',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   XLA_PYTHON_CLIENT_MEM_FRACTION='0.60',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        env.pop('XLA_PYTHON_CLIENT_ALLOCATOR', None)
        started = time.monotonic()
        proc = None
        reason = None
        interrupted = False

        def handle_signal(signum, frame):
            nonlocal interrupted
            interrupted = True

        old_handlers = {s: signal.signal(s, handle_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (output / 'supervised.log').open('w') as log:
                proc = subprocess.Popen(command, cwd=args.cwd, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                job['pid'] = proc.pid
                write_json(ledger_path, ledger)
                while proc.poll() is None:
                    elapsed = time.monotonic() - started
                    remaining = min(allowed - elapsed, deadline - time.time())
                    if interrupted or remaining <= 45:
                        reason = 'supervisor_interrupted' if interrupted else 'wall_or_calendar_limit'
                        stop_child(proc, max(0.1, min(40, remaining)))
                        break
                    job['elapsed_seconds'] = elapsed
                    job['updated_utc'] = utc_now()
                    write_json(ledger_path, ledger)
                    write_json(output / 'supervisor_status.json', job)
                    time.sleep(min(5, remaining))
                job['exit_code'] = proc.wait()
                job['state'] = 'complete' if proc.returncode == 0 and reason is None else 'stopped'
        except BaseException as exc:
            job['state'] = 'failed'
            job['error'] = f'{type(exc).__name__}: {exc}'
            if proc is not None:
                stop_child(proc, max(0.1, min(30, deadline - time.time(),
                                            allowed - (time.monotonic() - started))))
            raise
        finally:
            for s, handler in old_handlers.items():
                signal.signal(s, handler)
            job['elapsed_seconds'] = time.monotonic() - started
            job['finished_utc'] = utc_now()
            job['stop_reason'] = reason
            write_json(ledger_path, ledger)
            write_json(output / 'supervisor_status.json', job)
        return int(job.get('exit_code', 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--kind', choices=['training', 'evaluation', 'smoke'], required=True)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--total-budget-seconds', type=float, default=MAXIMUM_GPU_SECONDS,
                   help='total charged GPU wall budget, including evaluations; immutable per ledger')
    p.add_argument('--candidate-training-budget-seconds', type=float,
                   default=MAXIMUM_CANDIDATE_TRAINING_SECONDS,
                   help='cumulative training wall budget per candidate ID; immutable per ledger')
    p.add_argument('--maximum-candidates', type=int, default=MAXIMUM_CANDIDATES,
                   help='maximum distinct training candidate IDs; immutable per ledger')
    p.add_argument('--deadline-utc', default='2026-10-27T13:30:00Z')
    p.add_argument('--wait-queue', required=True)
    p.add_argument('--wait-seconds', type=float, default=0)
    p.add_argument('--output', required=True)
    p.add_argument('--cwd', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    raise SystemExit(run(p.parse_args()))


if __name__ == '__main__':
    main()
