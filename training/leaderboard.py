"""CPU-only Track 2 reporting in the challenge's column order and units.

These are local estimates against the explicitly named reference, not Kaggle
scores. This module intentionally needs only the standard library so the cloud
status endpoint and native JAX launcher can use it without importing PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

METRICS = (
    ("add_auc", "AUC ↑"),
    ("spider_sr", "SP-SR ↑"),
    ("maniptrans_sr", "MP-SR ↑"),
    ("mppe_cm", "MPPE (cm) ↓"),
)
METRIC_SOURCE = "https://nvidia-isaac.github.io/video_to_data/v2d_challenge/"


def leaderboard_scores(evaluation):
    """Map native/custom evaluations; missing or invalid values stay unavailable.

    CHORD reference-tracking success is not SP-SR or MP-SR. Its legacy mppe_cm
    reduces multiple bodies by max; Kaggle averages their six-keypoint errors.
    Prefer mean_per_point_error_cm and only fall back for a verified single body.
    """
    evaluation = evaluation or {}
    metrics = evaluation.get("metrics") or evaluation.get("evaluation") or evaluation
    if evaluation.get("leaderboard"):
        metrics = evaluation["leaderboard"]["scores"]
    elif metrics.get("leaderboard"):
        metrics = metrics["leaderboard"]["scores"]
    mppe = metrics.get("mean_per_point_error_cm")
    if mppe is None and (
        "spider_sr" in metrics or len(metrics.get("object_body_names", ())) == 1
    ):
        mppe = metrics.get("mppe_cm")
    candidates = {
        "add_auc": metrics.get("add_auc"),
        "spider_sr": metrics.get("spider_sr", metrics.get("spider_sr_uncentered")),
        "maniptrans_sr": metrics.get("maniptrans_sr"),
        "mppe_cm": mppe,
    }
    scores = {}
    for key, value in candidates.items():
        valid = isinstance(value, (float, int)) and not isinstance(value, bool)
        valid = valid and math.isfinite(value) and value >= 0
        valid = valid and (key == "mppe_cm" or value <= 1)
        scores[key] = float(value) if valid else None
    return scores


def leaderboard_row(
    evaluation,
    *,
    tier,
    episode,
    variant,
    training_s=None,
    transitions=None,
    reference=None,
):
    metrics = evaluation.get("metrics") or evaluation.get("evaluation") or evaluation
    return {
        "tier": tier,
        "episode": episode,
        "variant": variant,
        **leaderboard_scores(evaluation),
        "world_count": metrics.get(
            "world_count", len(evaluation.get("valid_worlds", [])) or None
        ),
        "training_hours": training_s / 3600 if training_s is not None else None,
        "transitions": transitions,
        "metric_reference": reference
        or evaluation.get("metric_reference")
        or f"tier{tier}_input_reference",
        "official_leaderboard_score": False,
        "checkpoint_sha256": evaluation.get("checkpoint_sha256")
        or evaluation.get("provenance", {}).get("checkpoint_sha256"),
    }


def write_report(output, rows):
    """Write JSON, CSV and readable Markdown without changing source evaluations."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    report = {
        "schema": "v2d_track2_local_leaderboard_v1",
        "metric_source": METRIC_SOURCE,
        "metric_order": [key for key, _ in METRICS],
        "rate_scale": "0_to_1",
        "mppe_unit": "cm",
        "official_leaderboard_score": False,
        "scope": "Listed episodes only; no competition rank or 30-episode aggregate is inferred.",
        "auc_reference": "Local visible-mesh samples; the host's AUC sample set is not supplied.",
        "rows": rows,
    }
    columns = [
        "variant",
        "tier",
        "episode",
        *(key for key, _ in METRICS),
        "training_hours",
        "world_count",
        "transitions",
        "metric_reference",
    ]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    headers = [
        "Input",
        "Tier",
        "Episode",
        *(label for _, label in METRICS),
        "Training (h)",
        "Worlds",
    ]
    lines = [
        "Track 2 local evaluation — leaderboard format",
        "",
        "AUC and success rates use 0–1; MPPE is in cm. Missing values are unavailable, not zero.",
        "Scores use the named reference and local mesh samples; these are not official leaderboard scores.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        values = [
            str(row[k]).replace("|", "\\|") for k in ("variant", "tier", "episode")
        ]
        values += [
            "—" if row.get(k) is None else f"{row[k]:.4f}"
            for k in [*(key for key, _ in METRICS), "training_hours"]
        ]
        values += [str(row.get("world_count") or "—")]
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "Reference: " + ", ".join(sorted({r["metric_reference"] for r in rows})),
        "Coverage: "
        + ", ".join(
            f"T{t}: {len({r['episode'] for r in rows if r['tier'] == t})}/30 episodes"
            for t in sorted({r["tier"] for r in rows})
        ),
        "Separate variants are not averaged. CHORD reference-tracking success is a separate diagnostic.",
        "",
    ]
    for name, text in (
        ("leaderboard.json", json.dumps(report, indent=2, allow_nan=False) + "\n"),
        ("leaderboard.csv", csv_buffer.getvalue()),
        ("leaderboard.md", "\n".join(lines)),
    ):
        path = output / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", required=True, help="Saved paired-experiment results JSON"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = [
        leaderboard_row(
            run["evaluation"],
            tier=result["tier"],
            episode=result["episode"],
            variant=run["variant"],
            training_s=run.get("training_elapsed_s"),
            transitions=run.get("transitions"),
            reference=result["metric_reference"],
        )
        for run in result["runs"]
        if run.get("evaluation")
    ]
    if not rows:
        parser.error("No completed evaluations to report")
    write_report(args.output, rows)
    print((Path(args.output) / "leaderboard.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
