"""Offline preparation first; GPU work happens only on explicit train/profile/evaluate."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser(
        "inspect", help="verify and inventory the portable handoff"
    )
    inspect.add_argument("--bundle", required=True)
    prepare = sub.add_parser(
        "prepare", help="create portable, hashed per-episode training jobs"
    )
    prepare.add_argument("--bundle", required=True)
    prepare.add_argument("--urdf", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--tier", type=int, choices=[1, 2])
    prepare.add_argument("--episode", type=int)
    prepare.add_argument("--all", action="store_true")
    prepare.add_argument(
        "--variants",
        nargs="+",
        choices=["original", "corrected"],
        default=["corrected"],
    )
    for command in ("train", "profile"):
        run = sub.add_parser(command)
        run.add_argument("--job", required=True)
        run.add_argument("--runtime", required=True)
        run.add_argument("--output", required=True)
        run.add_argument("--config")
        run.add_argument("--algorithm", choices=["sac", "model_based"])
        run.add_argument("--worlds", type=int)
        run.add_argument("--device")
        run.add_argument("--transitions", type=int)
        run.add_argument("--hours", type=float)
        run.add_argument("--seed", type=int)
        if command == "train":
            initialization = run.add_mutually_exclusive_group()
            initialization.add_argument("--resume")
            initialization.add_argument("--warm-model")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--job", required=True)
    evaluate.add_argument("--runtime", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--worlds", type=int, default=1)
    evaluate.add_argument("--device", default="cuda:0")
    plan = sub.add_parser("plan", help="write an experiment plan without starting jobs")
    plan.add_argument("--jobs", required=True)
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument(
        "--experiment",
        default=str(
            Path(__file__).resolve().parent / "configs/rl_pilot_two.json"
        ),
        help="episode/tier/variant selection; defaults to the eight-run pilot",
    )
    plan.add_argument(
        "--algorithms",
        nargs="+",
        choices=["sac", "model_based"],
        default=None,
        help="defaults to the single learner in --config",
    )
    plan.add_argument("--seeds", nargs="+", type=int, default=[0])
    plan.add_argument(
        "--warm-models",
        action="store_true",
        help="transfer models within tier/variant/seed; changes the ablation",
    )
    batch = sub.add_parser(
        "batch",
        help="explicitly execute a bounded number of planned jobs in separate processes",
    )
    batch.add_argument("--plan", required=True)
    batch.add_argument("--runtime", required=True)
    batch.add_argument("--max-jobs", type=int, default=1)
    batch.add_argument("--start", type=int, default=0)
    batch.add_argument("--evaluate", action="store_true")
    batch.add_argument("--device", help="device for both training and evaluation")
    batch.add_argument("--shard-index", type=int, default=0)
    batch.add_argument("--shard-count", type=int, default=1)
    doctor = sub.add_parser(
        "doctor", help="check runtime, assets and GPU visibility; does not train"
    )
    doctor.add_argument("--runtime", required=True)
    return p


def make_plan(args):
    from .bundles import sha256
    from .config import TrainConfig
    from .prepare import load_job

    config = TrainConfig.load(args.config)
    experiment = json.loads(Path(args.experiment).read_text(encoding="utf-8"))
    selected = {
        (tier, episode, variant)
        for tier in experiment["tiers"]
        for episode in experiment["episodes"]
        for variant in experiment["variants"]
    }
    if not selected or any(
        t not in (1, 2) or v not in ("original", "corrected") for t, _, v in selected
    ):
        raise ValueError("experiment must select Tier 1/2 original/corrected jobs")
    algorithms = args.algorithms or [config.algorithm]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    relative = lambda path: Path(
        os.path.relpath(Path(path).resolve(), output.parent)
    ).as_posix()
    jobs = sorted(Path(args.jobs).resolve().rglob("job.json"))
    entries, previous, found = [], {}, set()
    for path in jobs:
        job = load_job(path.parent)
        identity = (job["tier"], job["episode_index"], job["variant"])
        if identity not in selected:
            continue
        if identity in found:
            raise ValueError(f"duplicate selected job: {identity}")
        found.add(identity)
        for algorithm in algorithms:
            for seed in args.seeds:
                run = (
                    Path("runs")
                    / f"tier{job['tier']}"
                    / job["variant"]
                    / algorithm
                    / f"episode_{job['episode_index']:06d}"
                    / f"seed_{seed}"
                )
                scope = (job["tier"], job["variant"], algorithm, seed)
                entry = {
                    "job": relative(path.parent),
                    "job_sha256": sha256(path),
                    "algorithm": algorithm,
                    "seed": seed,
                    "output": run.as_posix(),
                }
                if (
                    args.warm_models
                    and algorithm == "model_based"
                    and scope in previous
                ):
                    entry["warm_model"] = (
                        previous[scope] / "world_model.pt"
                    ).as_posix()
                previous[scope] = run
                entries.append(entry)
    if found != selected:
        raise ValueError(f"missing selected jobs: {sorted(selected - found)}")
    plan = {
        "schema": "v2d_training_plan_v1",
        "config": config.to_dict(),
        "entries": entries,
        "run_count": len(entries),
        "configured_max_training_hours": len(entries) * config.max_hours,
        "budget_note": "configured cap, not a prediction; initialization/evaluation are additional",
        "model_transfer": args.warm_models,
        "experiment": experiment,
    }
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {
        "plan": str(output),
        "runs": len(entries),
        "maximum_training_hours": plan["configured_max_training_hours"],
        "started": False,
    }


def batch_run(args):
    from .bundles import sha256
    from .prepare import load_job

    if (
        args.max_jobs <= 0
        or args.start < 0
        or args.shard_count < 1
        or not 0 <= args.shard_index < args.shard_count
    ):
        raise ValueError("invalid batch bounds")
    path = Path(args.plan).resolve()
    plan = json.loads(path.read_text())
    if plan.get("schema") != "v2d_training_plan_v1":
        raise ValueError("unknown plan schema")
    if plan.get("obsolete"):
        raise ValueError(
            "obsolete experiment plan; generate the current two-episode pilot"
        )
    if args.shard_count > 1 and plan.get("model_transfer"):
        raise ValueError(
            "independent shards cannot preserve sequential world-model transfer"
        )
    selected_entries = plan["entries"][args.shard_index :: args.shard_count]
    completed = []
    for entry in selected_entries[args.start : args.start + args.max_jobs]:
        job, output = path.parent / entry["job"], path.parent / entry["output"]
        load_job(job)
        if sha256(job / "job.json") != entry["job_sha256"]:
            raise ValueError("job changed after planning")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"planned run already has output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        config = output / "planned_config.json"
        values = dict(plan["config"])
        if args.device:
            values["device"] = args.device
        config.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "training",
            "train",
            "--job",
            str(job),
            "--runtime",
            str(Path(args.runtime).resolve()),
            "--output",
            str(output),
            "--config",
            str(config),
            "--algorithm",
            entry["algorithm"],
            "--seed",
            str(entry["seed"]),
        ]
        if entry.get("warm_model"):
            command += ["--warm-model", str(path.parent / entry["warm_model"])]
        subprocess.run(command, check=True)
        if args.evaluate:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training",
                    "evaluate",
                    "--job",
                    str(job),
                    "--runtime",
                    str(Path(args.runtime).resolve()),
                    "--checkpoint",
                    str(output / "latest.pt"),
                    "--output",
                    str(output / "evaluation"),
                    "--device",
                    values["device"],
                ],
                check=True,
            )
        completed.append(entry["output"])
    return {"completed": completed}


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "inspect":
        from .bundles import inventory

        result = inventory(args.bundle)
    elif args.command == "prepare":
        from .prepare import prepare_episode

        if args.all:
            if args.tier is not None or args.episode is not None:
                p.error("--all is mutually exclusive with --tier/--episode")
            index = json.loads((Path(args.bundle) / "manifest.json").read_text())
            pairs = [
                (e["tier"], e["episode_index"])
                for e in index["entries"]
                if e["tier"] in (1, 2)
            ]
        else:
            if args.tier is None or args.episode is None:
                p.error("provide --all or both --tier and --episode")
            pairs = [(args.tier, args.episode)]
        result = []
        for tier, episode in pairs:
            for variant in args.variants:
                output = (
                    Path(args.output)
                    / f"tier{tier}"
                    / f"episode_{episode:06d}"
                    / variant
                )
                job = prepare_episode(
                    args.bundle, tier, episode, variant, args.urdf, output
                )
                result.append(
                    {
                        "path": str(output),
                        "tier": tier,
                        "episode": episode,
                        "variant": variant,
                        "frames": job["source_frames"],
                    }
                )
    elif args.command in ("train", "profile"):
        from .config import TrainConfig
        from .runner import train
        from .runtime import SonicEnv

        overrides = {
            "algorithm": args.algorithm,
            "worlds": args.worlds,
            "device": args.device,
            "seed": args.seed,
            "total_transitions": args.transitions,
            "max_hours": args.hours,
        }
        if args.command == "profile":
            overrides.update(
                max_hours=args.hours if args.hours is not None else 0.1,
                total_transitions=args.transitions
                if args.transitions is not None
                else 250000,
            )
        config = TrainConfig.load(args.config, **overrides)
        if (Path(args.output) / "run.json").exists():
            raise FileExistsError(
                "choose a new run directory before starting the simulator"
            )
        if config.algorithm == "model_based" and config.worlds < 2:
            raise ValueError(
                "model training needs at least two worlds for held-out validation"
            )
        env = SonicEnv(args.job, args.runtime, config, Path(args.output) / "runtime")
        result = train(
            env,
            config,
            args.output,
            getattr(args, "resume", None),
            getattr(args, "warm_model", None),
        )
        if args.command == "profile":
            result["profile_note"] = (
                "Includes learner work after warmup; extrapolate only a steady-state metrics window. No 90-task runtime claim."
            )
    elif args.command == "evaluate":
        from .checkpoint import read_checkpoint
        from .config import TrainConfig
        from .evaluate import evaluate
        from .runtime import SonicEnv

        values = read_checkpoint(args.checkpoint)["config"]
        values.update(worlds=args.worlds, device=args.device)
        config = TrainConfig(**values).validate()
        if any(
            (Path(args.output) / name).exists()
            for name in ("evaluation.json", "object_trajectories.parquet")
        ):
            raise FileExistsError(
                "choose a new evaluation directory before starting the simulator"
            )
        env = SonicEnv(args.job, args.runtime, config, Path(args.output) / "runtime")
        result = evaluate(env, args.checkpoint, args.output)
    elif args.command == "plan":
        result = make_plan(args)
    elif args.command == "batch":
        result = batch_run(args)
    else:
        import importlib.util

        import torch

        from . import RUNTIME_COMMIT
        from .bundles import sha256
        from .runtime import runtime_source

        source = runtime_source(args.runtime)
        asset = source / "flash_chord/assets"
        files = {
            p.relative_to(asset).as_posix(): {
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for p in (asset / "policies/sonic").glob("*.onnx")
        }
        result = {
            "runtime_commit": RUNTIME_COMMIT,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dependencies": {
                k: importlib.util.find_spec(k) is not None
                for k in ("warp", "newton", "onnxruntime", "hydra", "pyarrow")
            },
            "sonic_assets": files,
            "reminder": "LFS pointer files are not ONNX weights; construction verifies hashes",
        }
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
