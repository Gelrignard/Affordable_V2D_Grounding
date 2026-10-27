"""Native acquisition and precision-refinement configuration contracts."""
import bisect
import copy
import json
from pathlib import Path

DEFAULTS = json.loads((Path(__file__).resolve().parents[1] / "configs/policy.json").read_text(encoding="utf-8"))
WORLDS = DEFAULTS["worlds"]
REPLAY_CAPACITY = DEFAULTS["replay_capacity"]
OPTIMIZER_HORIZON = DEFAULTS["optimizer_horizon"]


def normalize_checkpoint_config(saved):
    """Translate historical serialized names, without importing retired code.

    The saved ramp origin is preserved. Only names change; checkpoint tensors,
    optimizer clocks, references, reward values and reset settings do not.
    """
    cfg = copy.deepcopy(saved)
    legacy_targets = {
        "training.overnight_refinement.OvernightObjectiveConfig": "training.core.refinement.RefinementObjectiveConfig",
        "training.overnight_refinement.OvernightTerminationConfig": "training.core.refinement.RefinementTerminationConfig",
    }
    def visit(value):
        if isinstance(value, dict):
            if value.get("_target_") in legacy_targets:
                value["_target_"] = legacy_targets[value["_target_"]]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(cfg)
    if "overnight" in cfg:
        old = cfg.pop("overnight")
        if "refinement" in cfg and cfg["refinement"] != old:
            raise ValueError("Conflicting historical and current refinement metadata")
        cfg["refinement"] = old
    return cfg


def refinement_curriculum(objective, start, end, phase, ramp_transitions=10_000_000):
    weights = {name.removesuffix("_weight"): value for name, value in objective.items()
               if name.endswith("_weight")}
    weights["object_orientation"] = 0.0
    fractions = [1.0] if phase == "consolidate" else [i / 10 for i in range(11)]
    interval = max(WORLDS, (ramp_transitions // 10 // WORLDS) * WORLDS)
    stages = []
    for fraction in fractions:
        stage_weights = {**weights, "object_position": 1.0 + 2.0 * fraction,
                         "force_closure": 5.0 - 4.0 * fraction,
                         "object_orientation": 2.0 * fraction}
        stages.append({"_target_": "flash_chord.configuration.make_curriculum_stage",
                       "voc_scale": 0.0, "objective_weights": stage_weights,
                       "immediate_first_frame_probability": 1.0 if phase == "consolidate" else 0.5})
    thresholds = ([end] if phase == "consolidate" else
                  [start + interval * (i + 1) for i in range(10)] + [max(end, start + 11 * interval)])
    return {"_target_": "flash_chord.configuration.make_fixed_curriculum",
            "thresholds": thresholds, "stages": stages}


def build_config(saved, *, state, output, start, end, phase, checkpoint_interval):
    """Preserve saved network, observation, action, source and simulator contracts."""
    cfg = normalize_checkpoint_config(saved)
    if int(cfg["scene"]["world_count"]) != WORLDS:
        raise ValueError("Native compact warm start requires a 2048-world source checkpoint")
    objective = copy.deepcopy(cfg["env"]["objective"])
    objective.update({"_target_": "training.core.refinement.RefinementObjectiveConfig",
                      "object_orientation_weight": 0.0 if phase == "refine" else 2.0,
                      "object_orientation_std": 0.6, "contact_blend_frames": 10,
                      "hand_position_tolerance": 0.03, "relative_hand_position_std": 0.15,
                      "hand_orientation_std": 0.8, "hand_orientation_factor": 0.25,
                      "object_position_fine_fraction": 1.0 / 3.0,
                      "object_position_fine_std": 0.05,
                      "use_object_relative_hands": True})
    objective["object_position_weight"] = 1.0 if phase == "refine" else 3.0
    objective["force_closure_weight"] = 5.0 if phase == "refine" else 1.0
    termination = copy.deepcopy(cfg["env"]["termination"])
    termination.update({"_target_": "training.core.refinement.RefinementTerminationConfig",
                        "palm_position_threshold": 0.25, "palm_orientation_threshold": 2.2,
                        "object_position_threshold": 0.20, "palm_dwell_steps": 10,
                        "object_dwell_steps": 5, "palm_gross_position": 0.60,
                        "palm_gross_orientation": 2.8, "object_gross_position": 0.40,
                        "object_gross_orientation": 2.8})
    reset = copy.deepcopy(cfg["env"]["reset"])
    reset.update({"reset_voc_scale": 0.0 if phase == "consolidate" else 1.0,
                  "voc_decay_steps": 0 if phase == "consolidate" else 10,
                  "always_reset_to_first_frame": phase == "consolidate",
                  "reset_to_first_frame_probability": 0.0,
                  "immediate_first_frame_probability": 1.0 if phase == "consolidate" else 0.5})
    if phase == "consolidate":
        reset.update(reset_freeze_steps=0, shoulder_spread=0.0)
        termination["reset_freeze_steps"] = 0
    else:
        reset["reset_freeze_steps"] = 50
        termination["reset_freeze_steps"] = 50
    for name, value in (("objective", objective), ("termination", termination), ("reset", reset)):
        cfg[name] = copy.deepcopy(value)
        cfg["env"][name] = value
    cfg["env"]["voc_scale"] = 0.0
    cfg["voc_scale_override"] = 0.0
    cfg["scene"]["world_count"] = WORLDS
    prior = cfg.get("refinement", {})
    ramp_start = int(prior.get("refinement_start_steps", start))
    if ramp_start > start:
        raise ValueError("Refinement ramp origin is after the restored checkpoint")
    if prior.get("phase") == "consolidate" and phase == "refine":
        raise ValueError("Cannot restart refinement after exact-start consolidation")
    curriculum = refinement_curriculum(objective, ramp_start, end, phase)
    active_index = min(bisect.bisect_right(curriculum["thresholds"], start), len(curriculum["stages"]) - 1)
    for name, weight in curriculum["stages"][active_index]["objective_weights"].items():
        cfg["env"]["objective"][name + "_weight"] = weight
        cfg["objective"][name + "_weight"] = weight
    cfg["refinement"] = {"phase": phase, "refinement_start_steps": ramp_start,
                        "method": "candidate2_position_precision_mixture",
                        "fresh_replay": True, "optimizer_horizon": OPTIMIZER_HORIZON,
                        "replay_capacity": REPLAY_CAPACITY, "replay_backend": "gpu",
                        "recovery_reset_assistance": phase == "refine"}
    cfg["curriculum"] = curriculum
    training = cfg["training"]
    training.update(world_count=WORLDS, total_environment_steps=end, resume=True,
                    checkpoint_path=str(state), curriculum=copy.deepcopy(curriculum))
    training["replay"].update(capacity=REPLAY_CAPACITY, batch_size=WORLDS,
                              clear_on_stage_change=False)
    training.setdefault("exploration", {})["warmup_action"] = "policy"
    training["optimization"]["learning_rate_schedule_updates"] = OPTIMIZER_HORIZON
    training["checkpoint"].update(save_interval_environment_steps=checkpoint_interval,
                                  save_replay=False, load_replay_path=None,
                                  load_optimizer=True, load_reward_normalizer=True)
    cfg["logging"].update(mode="offline", log_interval=25,
                          upload_checkpoints=False, upload_training_state=False)
    cfg["output_dir"] = str(output / "checkpoints")
    return cfg


def build_fresh_config(saved, *, output, end, checkpoint_interval):
    """Bound a resolved native bootstrap without changing its method or curriculum."""
    cfg = normalize_checkpoint_config(saved)
    if not isinstance(cfg, dict):
        raise ValueError("Fresh config must be a resolved JSON object")
    training = cfg.get("training", {})
    task = cfg.get("task", {})
    if cfg.get("refinement") or "reference" in cfg:
        raise ValueError("Fresh native config must not contain refinement or checkpoint reference state")
    if training.get("resume") is not False or training.get("checkpoint_path") is not None:
        raise ValueError("Fresh native config requires resume=false and checkpoint_path=null")
    if training.get("checkpoint", {}).get("load_replay_path") is not None:
        raise ValueError("Fresh native config cannot restore replay")
    if cfg.get("scene", {}).get("world_count") != WORLDS or training.get("world_count") != WORLDS:
        raise ValueError("Fresh native bootstrap requires 2048 scene and training worlds")
    if training.get("seed") != 42 or cfg.get("env", {}).get("reset", {}).get("seed") != 42:
        raise ValueError("Fresh native bootstrap requires the native seed 42")
    if task.get("source_frame_playback") is not True or task.get("motion_start_frame") != 0 or task.get("motion_end_frame") != -1:
        raise ValueError("Fresh native bootstrap requires the full source-frame sequence")
    parquet = task.get("parquet")
    if not isinstance(parquet, str) or not Path(parquet).is_absolute() or not Path(parquet).is_file():
        raise ValueError("Fresh native task parquet must be an existing absolute path")
    if cfg.get("env", {}).get("objective", {}).get("_target_") != "flash_chord.objectives.recon_body.ReconBodyObjectiveConfig":
        raise ValueError("Fresh bootstrap requires the native ReconBody objective")
    curriculum = cfg.get("curriculum")
    if not isinstance(curriculum, dict) or training.get("curriculum") != curriculum:
        raise ValueError("Fresh native curriculum must be resolved and identical in training")
    thresholds = [35000320, 70000640, 104998912, 139999232, 174999552,
                  209999872, 288749568, 306249728, 349999104]
    stages = curriculum.get("stages", [])
    if curriculum.get("thresholds") != thresholds or [stage.get("voc_scale") for stage in stages] != [1., .75, .5, .25, .1, .05, 0., 0., 0.]:
        raise ValueError("Fresh bootstrap must retain the native 350M VOC curriculum")
    if cfg.get("voc_scale_override") is not None:
        raise ValueError("Fresh bootstrap must not override the native VOC curriculum")
    if end <= 0 or end % WORLDS or end > thresholds[-1]:
        raise ValueError("Fresh bootstrap transition cap must align with worlds and not exceed native 350M")
    if training.get("optimization", {}).get("learning_rate_schedule_updates") != OPTIMIZER_HORIZON:
        raise ValueError("Fresh bootstrap must retain the native 350M optimizer horizon")
    training.update(total_environment_steps=end, resume=False, checkpoint_path=None)
    training["replay"].update(capacity=REPLAY_CAPACITY, batch_size=WORLDS)
    training["checkpoint"].update(save_interval_environment_steps=checkpoint_interval,
                                  save_replay=False, load_replay_path=None,
                                  load_optimizer=False, load_reward_normalizer=False)
    cfg["logging"].update(mode="offline", log_interval=25,
                          upload_checkpoints=False, upload_training_state=False)
    cfg["output_dir"] = str(output / "checkpoints")
    return cfg
