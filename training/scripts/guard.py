"""Standard-library cloud supervisor: serialized jobs and durable wall budgets.

This module never imports CUDA. It never stops jobs it did not start. A blocked
queue is not a GPU run and is not charged to the additional-GPU-time budget.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
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
    if value is None:
        return math.inf
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


def make_budget_policy(total_budget_seconds=None, candidate_training_budget_seconds=None,
                       maximum_candidates=None):
    """Optional limits for new runs; None is unbounded and remains JSON null."""
    for name, value in [('total_budget_seconds', total_budget_seconds),
                        ('candidate_training_budget_seconds', candidate_training_budget_seconds)]:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be None or a positive finite number')
    if maximum_candidates is not None and (isinstance(maximum_candidates, bool)
            or not isinstance(maximum_candidates, int) or maximum_candidates < 1):
        raise ValueError('maximum_candidates must be None or a positive integer')
    finite = all(value is not None for value in
                 (total_budget_seconds, candidate_training_budget_seconds, maximum_candidates))
    return {'schema': 'v2d_gpu_budget_policy_v1' if finite else 'v2d_gpu_budget_policy_v2',
            'total_budget_seconds': None if total_budget_seconds is None else float(total_budget_seconds),
            'candidate_training_budget_seconds': None if candidate_training_budget_seconds is None
                                                  else float(candidate_training_budget_seconds),
            'maximum_candidates': maximum_candidates}


def _validated_policy(policy):
    if not isinstance(policy, dict) or set(policy) != set(make_budget_policy()):
        raise RuntimeError('Ledger budget policy is malformed; fail closed')
    try:
        canonical = make_budget_policy(policy['total_budget_seconds'],
                                       policy['candidate_training_budget_seconds'],
                                       policy['maximum_candidates'])
    except ValueError as exc:
        raise RuntimeError('Ledger budget policy is invalid; fail closed') from exc
    if policy != canonical:
        raise RuntimeError('Ledger budget policy schema is unsupported; fail closed')
    return canonical


def ledger_budget_policy(ledger):
    """Original policy. Policy-less historical ledgers keep the old finite limits."""
    if 'budget_policy' not in ledger:
        return make_budget_policy(MAXIMUM_GPU_SECONDS, MAXIMUM_CANDIDATE_TRAINING_SECONDS,
                                  MAXIMUM_CANDIDATES)
    return _validated_policy(ledger['budget_policy'])


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def effective_budget_policy(ledger, candidate=None, *, ledger_path=None):
    """Follow explicit policy changes without discarding any historical charge."""
    original = ledger_budget_policy(ledger)
    policy = original
    updates = ledger.get('budget_policy_updates', [])
    if not isinstance(updates, list) or ledger.get('budget_amendments'):
        raise RuntimeError('Unsupported budget policy history')
    previous, count = None, 0
    fields = {'schema', 'updated_utc', 'ledger_path', 'prior_ledger_sha256',
              'original_policy_sha256', 'history_job_count', 'history_jobs_sha256',
              'previous_update_sha256', 'policy', 'authorization_reason', 'record_sha256'}
    for record in updates:
        if not isinstance(record, dict) or set(record) != fields or record['schema'] != 'v2d_gpu_budget_policy_update_v1':
            raise RuntimeError('Budget policy update is malformed')
        body = {key: value for key, value in record.items() if key != 'record_sha256'}
        if canonical_sha256(body) != record['record_sha256']:
            raise RuntimeError('Budget policy update hash differs')
        if (record['previous_update_sha256'] != previous
                or record['original_policy_sha256'] != canonical_sha256(original)):
            raise RuntimeError('Budget policy chain or original policy differs')
        if not isinstance(record['authorization_reason'], str) or not record['authorization_reason'].strip():
            raise RuntimeError('Budget policy update requires an authorization reason')
        prior_hash = record['prior_ledger_sha256']
        if (not isinstance(prior_hash, str) or len(prior_hash) != 64
                or any(c not in '0123456789abcdef' for c in prior_hash)):
            raise RuntimeError('Invalid prior ledger hash')
        if not isinstance(record['ledger_path'], str) or not Path(record['ledger_path']).is_absolute():
            raise RuntimeError('Budget policy update requires an absolute ledger path')
        if ledger_path is not None and Path(record['ledger_path']).resolve() != Path(ledger_path).resolve():
            raise RuntimeError('Budget policy update belongs to another canonical ledger')
        next_count = record['history_job_count']
        jobs = ledger.get('jobs', [])
        if (isinstance(next_count, bool) or not isinstance(next_count, int)
                or not count <= next_count <= len(jobs)):
            raise RuntimeError('Budget policy historical prefix length differs')
        prefix = jobs[:next_count]
        if (canonical_sha256(prefix) != record['history_jobs_sha256']
                or any(row.get('state') not in {'complete', 'stopped', 'failed'} for row in prefix)):
            raise RuntimeError('Budget policy historical job rows changed or are not terminal')
        policy = _validated_policy(record['policy'])
        previous, count = record['record_sha256'], next_count
    return dict(policy)


def ensure_budget_policy(ledger, requested_policy, *, ledger_exists):
    """Pin new ledgers. Omitted flags inherit existing effective limits.

    Removing or changing old limits requires set-budget; omission never releases
    a finite historical cap. An explicit policy update leaves the base untouched.
    """
    requested = _validated_policy(requested_policy)
    if ledger_exists or 'budget_policy' in ledger:
        existing = effective_budget_policy(ledger)
        for key in ('total_budget_seconds', 'candidate_training_budget_seconds', 'maximum_candidates'):
            if requested[key] is not None and requested[key] != existing[key]:
                raise RuntimeError('Requested budget differs from immutable ledger policy; '
                                   'use set-budget with an explicit authorization reason')
        if 'budget_policy' not in ledger:
            ledger['budget_policy'] = ledger_budget_policy(ledger)
        return existing
    ledger['budget_policy'] = requested
    return requested


def set_budget_policy(ledger_path, policy, authorization_reason):
    """Explicit, locked policy change; omitted CLI limits here mean unbounded."""
    ledger_path = Path(ledger_path).resolve()
    policy = _validated_policy(policy)
    if not isinstance(authorization_reason, str) or not authorization_reason.strip():
        raise ValueError('An explicit authorization reason is required')
    with exclusive_lock(ledger_path.with_suffix('.lock')):
        original_bytes = ledger_path.read_bytes()
        ledger = json.loads(original_bytes)
        existing = effective_budget_policy(ledger, ledger_path=ledger_path)
        jobs = ledger.get('jobs', [])
        if any(job.get('state') not in {'complete', 'stopped', 'failed'} for job in jobs):
            raise RuntimeError('Cannot change policy while a job is running or unresolved')
        if any(pid_live(job.get(key)) for job in jobs for key in ('pid', 'child_pid', 'supervisor_pid')):
            raise RuntimeError('Cannot change policy with a live or reused worker PID')
        if existing == policy:
            return {'state': 'unchanged', 'ledger': str(ledger_path),
                    'ledger_sha256': hashlib.sha256(original_bytes).hexdigest(), 'effective_policy': existing}
        updates = ledger.get('budget_policy_updates', [])
        record = {'schema': 'v2d_gpu_budget_policy_update_v1', 'updated_utc': utc_now(),
                  'ledger_path': str(ledger_path), 'prior_ledger_sha256': hashlib.sha256(original_bytes).hexdigest(),
                  'original_policy_sha256': canonical_sha256(ledger_budget_policy(ledger)),
                  'history_job_count': len(jobs), 'history_jobs_sha256': canonical_sha256(jobs),
                  'previous_update_sha256': updates[-1]['record_sha256'] if updates else None,
                  'policy': policy, 'authorization_reason': authorization_reason}
        record['record_sha256'] = canonical_sha256(record)
        ledger.setdefault('budget_policy_updates', []).append(record)
        effective_budget_policy(ledger, ledger_path=ledger_path)
        write_json(ledger_path, ledger)
        return {'state': 'updated', 'ledger': str(ledger_path), 'prior_ledger_sha256': record['prior_ledger_sha256'],
                'ledger_sha256': hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                'update_sha256': record['record_sha256'], 'effective_policy': policy}


def remaining_budget(ledger, candidate, requested_seconds=None, kind='training'):
    jobs = ledger.get('jobs', [])
    if any(j.get('state') == 'running' for j in jobs):
        raise RuntimeError('Ledger contains a running job; reconcile its PID before launching')
    policy = effective_budget_policy(ledger, candidate)
    elapsed = [float(j.get('elapsed_seconds', 0)) for j in jobs]
    if any(not math.isfinite(value) or value < 0 for value in elapsed):
        raise RuntimeError('Ledger elapsed charges must be finite and nonnegative')
    candidates = {j['candidate'] for j in jobs if j.get('kind') == 'training'}
    maximum = policy['maximum_candidates']
    if kind == 'training' and candidate not in candidates and maximum is not None and len(candidates) >= maximum:
        raise RuntimeError(f'Candidate implementation limit ({maximum}) reached')
    available = math.inf if policy['total_budget_seconds'] is None else policy['total_budget_seconds'] - sum(elapsed)
    if kind == 'training' and policy['candidate_training_budget_seconds'] is not None:
        trained = sum(value for j, value in zip(jobs, elapsed)
                      if j.get('candidate') == candidate and j.get('kind') == 'training')
        available = min(available, policy['candidate_training_budget_seconds'] - trained)
    requested = math.inf if requested_seconds is None else float(requested_seconds)
    if math.isnan(requested) or requested < 0:
        raise ValueError('Requested budget must be nonnegative')
    return max(0.0, min(requested, available))


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
    if queue is None:
        pids = gpu_pids()
        return (False, f'Other GPU processes still present: {pids}') if pids else (True, 'GPU idle')
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
            raise RuntimeError('Another policy worker owns the GPU lock') from exc
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
            getattr(args, 'total_budget_seconds', None),
            getattr(args, 'candidate_training_budget_seconds', None),
            getattr(args, 'maximum_candidates', None))
        ensure_budget_policy(ledger, requested_policy, ledger_exists=ledger_exists)
        effective_budget_policy(ledger, args.candidate, ledger_path=ledger_path)
        allowed = remaining_budget(ledger, args.candidate, args.max_seconds, args.kind)
        if allowed <= 0 or deadline - time.time() <= 0:
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
        if allowed <= 0:
            raise RuntimeError('Insufficient wall/calendar budget after waiting')
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        job = {'name': args.name, 'candidate': args.candidate, 'kind': args.kind,
               'state': 'running', 'started_utc': utc_now(), 'elapsed_seconds': 0.0,
               'maximum_seconds': None if math.isinf(allowed) else allowed, 'deadline_utc': args.deadline_utc,
               'command': command, 'output': str(output), 'supervisor_pid': os.getpid()}
        ledger['jobs'].append(job)
        write_json(ledger_path, ledger)
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', V2D_POLICY_GUARD='1',
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['set-budget']:
        p = argparse.ArgumentParser(description='Explicitly change effective limits, preserving old policy and charges. Omitted limits become unbounded.')
        p.add_argument('--ledger', required=True)
        p.add_argument('--authorization-reason', required=True)
        p.add_argument('--total-budget-seconds', type=float)
        p.add_argument('--candidate-training-budget-seconds', type=float)
        p.add_argument('--maximum-candidates', type=int)
        args = p.parse_args(argv[1:])
        result = set_budget_policy(args.ledger, make_budget_policy(args.total_budget_seconds,
                                   args.candidate_training_budget_seconds, args.maximum_candidates),
                                   args.authorization_reason)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        raise SystemExit(0)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--kind', choices=['training', 'evaluation', 'smoke'], required=True)
    p.add_argument('--max-seconds', type=float, help='Optional wall limit for this command')
    p.add_argument('--total-budget-seconds', type=float,
                   help='Optional total charged wall limit, including evaluations; omitted flags inherit old ledger limits')
    p.add_argument('--candidate-training-budget-seconds', type=float,
                   help='Optional cumulative training wall limit per candidate ID')
    p.add_argument('--maximum-candidates', type=int, help='Optional distinct training candidate limit')
    p.add_argument('--deadline-utc', help='Optional calendar deadline with explicit UTC offset')
    readiness = p.add_mutually_exclusive_group()
    readiness.add_argument('--wait-queue', help='Actual existing queue to wait for, when requested')
    readiness.add_argument('--idle-only', action='store_true', help='Check GPU idleness directly (default); no queue sentinel')
    p.add_argument('--wait-seconds', type=float, default=0)
    p.add_argument('--output', required=True)
    p.add_argument('--cwd', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        p.error('--max-seconds must be positive and finite when supplied')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
