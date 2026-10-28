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
    parser.add_argument("--variants", nargs="+", choices=("auto", "original", "corrected"), default=["auto"],
                        help="auto prefers an available corrected reference, otherwise warns and uses original")
    args = parser.parse_args(argv)
    if "auto" in args.variants and len(args.variants) != 1:
        parser.error("--variants auto cannot be combined with explicit variants")
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
    from training.core.input_selection import input_selection_record, select_bundle_variant, warn_input_selection
    results = []
    for tier, episode in pairs:
        variants = ([select_bundle_variant(args.bundle, tier, episode)]
                    if args.variants == ["auto"] else args.variants)
        for variant in variants:
            output = Path(args.output) / f"tier{tier}" / f"episode_{episode:06d}" / variant
            job = prepare_episode(args.bundle, tier, episode, variant, args.urdf, output)
            selection = input_selection_record(output)
            if args.variants == ["auto"] and variant == "original":
                selection["selection_reason"] = "corrected_unavailable_using_original"
            warn_input_selection(selection)
            # Separate provenance keeps job bytes and its existing hash contract
            # unchanged, including any false/unknown physical/planning flags.
            (output / "input_selection.json").write_text(
                json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            results.append({"path": str(output.resolve()), "tier": tier, "episode": episode,
                            "variant": variant, "frames": job["source_frames"]})
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
