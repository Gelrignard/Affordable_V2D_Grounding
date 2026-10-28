"""Prepare, train, evaluate, select and export available episode artifacts.

Accept one portable job/episode directory, a root of prepared jobs, or one
episode from a correction bundle. This command does not reconstruct video.
Run it with the pinned native environment after installing the runtime/assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--job", help="Portable variant directory, or episode directory containing corrected/original")
    source.add_argument("--jobs-root", help="Prepared job root; run a deterministic serial batch or shard")
    source.add_argument("--bundle", help="Correction bundle containing a validated available episode reference")
    p.add_argument("--tier", type=int, choices=(1, 2, 3), help="Required with --bundle")
    p.add_argument("--episode", type=int, choices=range(30), help="Required with --bundle")
    p.add_argument("--urdf", help="Pinned G1/Dex3 URDF; required with --bundle")
    for name in ("runtime", "output", "deadline-utc", "submission-kit"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--dataset-input", "--dataset-root", required=True, dest="dataset_input",
                   help="Released tier input directory for one job; root containing tier1/2/3 for a batch")
    p.add_argument("--parent-state", help="Compatible native learner state for this exact single episode; otherwise initialize natively")
    p.add_argument("--recipe", choices=("auto", "legacy", "tier1-precision", "tier23-load-transfer"), default="auto")
    p.add_argument("--selection-policy", choices=("auto", "success", "tracking"), default="auto")
    p.add_argument("--training-budget-seconds", type=int, default=10800)
    p.add_argument("--total-budget-seconds", "--episode-total-budget-seconds", type=int, default=11700, dest="total_budget_seconds")
    p.add_argument("--gpu-hours", type=float, help="Required for --jobs-root; total charged GPU wall hours for the shard")
    p.add_argument("--tiers", type=int, choices=(1, 2, 3), nargs="+", help="Batch tier filter (default: all tiers)")
    p.add_argument("--episodes", type=int, choices=range(30), nargs="+", help="Batch episode filter")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--max-jobs", type=int)
    p.add_argument("--require-complete", action="store_true", help="Batch: require every requested identity before GPU work")
    p.add_argument("--prepare-only", action="store_true", help="Single episode: CPU native/input/export preflight without training")
    p.add_argument("--dry-run", action="store_true", help="Single episode: CPU preflight; batch: validated roster preview only")
    return p


def validate_arguments(args, p):
    if args.jobs_root:
        if args.gpu_hours is None:
            p.error("--jobs-root requires --gpu-hours")
        if args.parent_state or args.tier is not None or args.episode is not None or args.urdf:
            p.error("--parent-state/--tier/--episode/--urdf apply only to a single episode")
        if args.prepare_only:
            p.error("Batch --dry-run validates the roster; use single-episode --prepare-only for native scene preflight")
    else:
        if (args.gpu_hours is not None or args.tiers is not None or args.episodes is not None
                or args.shard_index != 0 or args.shard_count != 1 or args.max_jobs is not None or args.require_complete):
            p.error("GPU-hour, tier/episode filters and shard options require --jobs-root")
        if args.bundle:
            if args.tier is None or args.episode is None or not args.urdf:
                p.error("--bundle requires --tier, --episode and --urdf")
        elif args.tier is not None or args.episode is not None or args.urdf:
            p.error("--tier, --episode and --urdf are inferred from --job; supply them only with --bundle")


def prepare_bundle_job(args):
    """Prepare once, or verify an identical preparation without overwriting it."""
    from training.core.input_selection import select_bundle_variant, input_selection_record
    from training.core.prepare import prepare_episode
    from training.scripts.pipeline import output_lock

    bundle, urdf = Path(args.bundle).resolve(), Path(args.urdf).resolve()
    variant = select_bundle_variant(bundle, args.tier, args.episode)
    root = Path(args.output).resolve() / "prepared_input"
    job = root / f"tier{args.tier}" / f"episode_{args.episode:06d}" / variant
    request = {"schema": "v2d_run_preparation_v1", "bundle": str(bundle),
               "bundle_manifest_sha256": digest(bundle / "manifest.json"),
               "urdf": str(urdf), "urdf_sha256": digest(urdf), "tier": args.tier,
               "episode_index": args.episode, "variant": variant,
               "preparer_sha256": digest(ROOT / "training/core/prepare.py")}
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root / "prepare.lock"):
        receipt = root / "preparation_receipt.json"
        if receipt.exists():
            previous = json.loads(receipt.read_text(encoding="utf-8"))
            if previous["request"] != request:
                raise ValueError("Bundle preparation inputs changed; use a different output directory")
            record = input_selection_record(job)
            if previous["job_sha256"] != record["job_sha256"]:
                raise ValueError("Prepared job changed since its recorded preparation")
        else:
            # prepare_episode refuses a nonempty destination, including an
            # interrupted preparation; no artifact is overwritten or relabeled.
            prepare_episode(bundle, args.tier, args.episode, variant, urdf, job)
            record = input_selection_record(job)
            value = {"request": request, "job": str(job), "job_sha256": record["job_sha256"]}
            with receipt.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        if variant == "original":
            record["selection_reason"] = "corrected_unavailable_using_original"
        return record


def single_command(args, selection):
    recipe = args.recipe
    if recipe == "auto":
        recipe = "tier1-precision" if selection["tier"] == 1 else "tier23-load-transfer"
    policy = args.selection_policy
    if policy == "auto":
        policy = "tracking" if recipe == "tier23-load-transfer" else "success"
    command = [sys.executable, "-m", "training.scripts.pipeline", "--job", selection["job"],
               "--runtime", str(Path(args.runtime).resolve()), "--output", str(Path(args.output).resolve()),
               "--deadline-utc", args.deadline_utc, "--training-budget-seconds", str(args.training_budget_seconds),
               "--total-budget-seconds", str(args.total_budget_seconds), "--recipe", recipe,
               "--selection-policy", policy, "--dataset-input", str(Path(args.dataset_input).resolve()),
               "--submission-kit", str(Path(args.submission_kit).resolve())]
    if args.parent_state:
        command.extend(["--parent-state", str(Path(args.parent_state).resolve())])
    if args.prepare_only or args.dry_run:
        command.append("--prepare-only")
    return command


def batch_command(args):
    command = [sys.executable, "-m", "training.scripts.batch", "--jobs", str(Path(args.jobs_root).resolve()),
               "--runtime", str(Path(args.runtime).resolve()), "--output", str(Path(args.output).resolve()),
               "--deadline-utc", args.deadline_utc, "--gpu-hours", str(args.gpu_hours),
               "--training-budget-seconds", str(args.training_budget_seconds),
               "--episode-total-budget-seconds", str(args.total_budget_seconds),
               "--recipe", args.recipe, "--selection-policy", args.selection_policy,
               "--dataset-root", str(Path(args.dataset_input).resolve()),
               "--submission-kit", str(Path(args.submission_kit).resolve()),
               "--shard-index", str(args.shard_index), "--shard-count", str(args.shard_count)]
    for name in ("tiers", "episodes"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name, *map(str, value)])
    if args.max_jobs is not None:
        command.extend(["--max-jobs", str(args.max_jobs)])
    if args.require_complete:
        command.append("--require-complete")
    if args.dry_run:
        command.append("--dry-run")
    return command


def record_input_selection(output, selection):
    """Pin the chosen variant before any child can prepare or train it."""
    from training.scripts.pipeline import immutable_json, output_lock
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root / "input_selection.lock"):
        immutable_json(root / "input_selection.json", selection)


def main(argv=None):
    p = parser(); args = p.parse_args(argv); validate_arguments(args, p)
    if args.jobs_root:
        command = batch_command(args)
    else:
        from training.core.input_selection import select_portable_input, warn_input_selection
        selection = prepare_bundle_job(args) if args.bundle else select_portable_input(args.job)
        record_input_selection(args.output, selection)
        warn_input_selection(selection)
        command = single_command(args, selection)
    # An absolute script invocation works from any directory. Child module
    # discovery and source identity always use this committed repository.
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
