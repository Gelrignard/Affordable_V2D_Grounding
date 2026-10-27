"""Prepare hash-verified per-episode jobs from a portable correction bundle."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tier", type=int, choices=(1, 2, 3))
    parser.add_argument("--episode", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=("original", "corrected"), default=["corrected"])
    args = parser.parse_args(argv)
    if args.all:
        if args.tier is not None or args.episode is not None:
            parser.error("--all is mutually exclusive with --tier/--episode")
        index = json.loads((Path(args.bundle) / "manifest.json").read_text(encoding="utf-8"))
        pairs = sorted({(item["tier"], item["episode_index"]) for item in index["entries"] if item["tier"] in (1, 2, 3)})
    else:
        if args.tier is None or args.episode is None:
            parser.error("provide --all or both --tier and --episode")
        pairs = [(args.tier, args.episode)]
    if not pairs:
        parser.error("bundle contains no supported episodes")
    from training.core.prepare import prepare_episode
    results = []
    for tier, episode in pairs:
        for variant in args.variants:
            output = Path(args.output) / f"tier{tier}" / f"episode_{episode:06d}" / variant
            job = prepare_episode(args.bundle, tier, episode, variant, args.urdf, output)
            results.append({"path": str(output.resolve()), "tier": tier, "episode": episode,
                            "variant": variant, "frames": job["source_frames"]})
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
