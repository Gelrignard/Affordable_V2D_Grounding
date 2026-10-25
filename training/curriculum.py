"""CHORD's G1 assistance schedule, expressed in simulator transitions.

The fractions follow the pinned recon_body_flash_sac.yaml. We retain our own
learner and soft-reference rewards; this is not a FlashSAC reproduction.
"""

CHORD_TRANSITIONS = 349999104
CHORD_THRESHOLDS = (
    35000320,
    70000640,
    104998912,
    139999232,
    174999552,
    209999872,
    288749568,
    306249728,
)
VOC_SCALES = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.0, 0.0, 0.0)
FIRST_FRAME_PROBABILITIES = (0.0,) * 7 + (0.25, 0.5)


def stage_at(transitions, budget=CHORD_TRANSITIONS):
    if transitions < 0 or budget <= 0:
        raise ValueError("invalid curriculum progress")
    index = sum(
        transitions * CHORD_TRANSITIONS >= threshold * budget
        for threshold in CHORD_THRESHOLDS
    )
    return {
        "stage": index,
        "assistance_scale": VOC_SCALES[index],
        "first_frame_probability": FIRST_FRAME_PROBABILITIES[index],
    }
