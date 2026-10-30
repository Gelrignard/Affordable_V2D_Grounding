"""Native acquisition and precision-refinement configuration contracts."""
import bisect
import copy
import hashlib
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


def refinement_curriculum(objective, start, end, phase, ramp_transitions=10_000_000, worlds=WORLDS):
    weights = {name.removesuffix("_weight"): value for name, value in objective.items()
               if name.endswith("_weight")}
    weights["object_orientation"] = 0.0
    fractions = [1.0] if phase == "consolidate" else [i / 10 for i in range(11)]
    interval = max(worlds, (ramp_transitions // 10 // worlds) * worlds)
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


def build_config(saved, *, state, output, start, end, phase, checkpoint_interval,
                 acquisition_guidance=None, acquisition_reference=None,
                 placement_guidance=None, placement_reference=None, placement_support=None):
    """Preserve saved network, observation, action, source and simulator contracts."""
    cfg = normalize_checkpoint_config(saved)
    if acquisition_guidance not in (None, "precision", "grasp-region"):
        raise ValueError("Unknown acquisition guidance method")
    if placement_guidance not in (None, "off", "supported-release"):
        raise ValueError("Unknown placement guidance method")
    worlds = cfg["training"]["world_count"]
    if type(worlds) is not int or worlds <= 0 or cfg["scene"]["world_count"] != worlds:
        raise ValueError("Training and scene world counts must agree")
    replay_capacity = cfg["training"]["replay"]["capacity"]
    horizon = cfg["training"]["optimization"]["learning_rate_schedule_updates"]
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
    if acquisition_guidance == "grasp-region":
        objective["acquisition_enabled"] = True
    elif acquisition_guidance == "precision" and "acquisition_enabled" in objective:
        objective["acquisition_enabled"] = False
    if objective.get("acquisition_enabled", False):
        reference = acquisition_reference or objective.get("acquisition_reference_path")
        if not reference or not Path(reference).is_file():
            raise ValueError("Grasp-region guidance requires its input reference.npz")
        with Path(reference).open("rb") as stream:
            reference_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        prior_sha = objective.get("acquisition_reference_sha256")
        if prior_sha and prior_sha != reference_sha:
            raise ValueError("Acquisition reference changed since the saved checkpoint")
        objective.update(acquisition_reference_path=str(Path(reference).resolve()),
                         acquisition_reference_sha256=reference_sha)
    if placement_guidance == "supported-release":
        objective["placement_enabled"] = True
    elif placement_guidance == "off" and "placement_enabled" in objective:
        objective["placement_enabled"] = False
    if objective.get("placement_enabled", False):
        if objective.get("acquisition_enabled", False):
            raise ValueError("Supported release must be evaluated with precision acquisition")
        for name, supplied in (("reference", placement_reference), ("support", placement_support)):
            source = supplied or objective.get(f"placement_{name}_path")
            if not source or not Path(source).is_file():
                raise ValueError(f"Supported release requires its input {name} file")
            with Path(source).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            previous_digest = objective.get(f"placement_{name}_sha256")
            if previous_digest and previous_digest != digest:
                raise ValueError(f"Placement {name} changed since the saved checkpoint")
            objective[f"placement_{name}_path"] = str(Path(source).resolve())
            objective[f"placement_{name}_sha256"] = digest
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
    cfg["scene"]["world_count"] = worlds
    prior = cfg.get("refinement", {})
    ramp_start = int(prior.get("refinement_start_steps", start))
    if ramp_start > start:
        raise ValueError("Refinement ramp origin is after the restored checkpoint")
    if prior.get("phase") == "consolidate" and phase == "refine":
        raise ValueError("Cannot restart refinement after exact-start consolidation")
    curriculum = refinement_curriculum(objective, ramp_start, end, phase, worlds=worlds)
    active_index = min(bisect.bisect_right(curriculum["thresholds"], start), len(curriculum["stages"]) - 1)
    for name, weight in curriculum["stages"][active_index]["objective_weights"].items():
        cfg["env"]["objective"][name + "_weight"] = weight
        cfg["objective"][name + "_weight"] = weight
    cfg["refinement"] = {"phase": phase, "refinement_start_steps": ramp_start,
                        "method": "candidate2_position_precision_mixture",
                        "fresh_replay": True, "optimizer_horizon": horizon,
                        "replay_capacity": replay_capacity, "replay_backend": "gpu",
                        "recovery_reset_assistance": phase == "refine"}
    if objective.get("acquisition_enabled", False):
        cfg["refinement"]["method"] = "object_attached_acquisition_region_v1"
        cfg["refinement"]["acquisition_reference_sha256"] = objective["acquisition_reference_sha256"]
    if objective.get("placement_enabled", False):
        cfg["refinement"]["method"] = "supported_load_transfer_v2"
        cfg["refinement"]["placement_reference_sha256"] = objective["placement_reference_sha256"]
        cfg["refinement"]["placement_support_sha256"] = objective["placement_support_sha256"]
    cfg["curriculum"] = curriculum
    training = cfg["training"]
    training.update(world_count=worlds, total_environment_steps=end, resume=True,
                    checkpoint_path=str(state), curriculum=copy.deepcopy(curriculum))
    training["replay"].update(clear_on_stage_change=False)
    training.setdefault("exploration", {})["warmup_action"] = "policy"
    training["optimization"]["learning_rate_schedule_updates"] = horizon
    training["checkpoint"].update(save_interval_environment_steps=checkpoint_interval,
                                  save_replay=False, load_replay_path=None,
                                  load_optimizer=True, load_reward_normalizer=True)
    cfg["logging"].update(mode="offline", log_interval=25,
                          upload_checkpoints=False, upload_training_state=False)
    cfg["output_dir"] = str(output / "checkpoints")
    return cfg


def validate_native_config(cfg):
    """Validate compatible native structure without imposing one recipe's defaults."""
    if not isinstance(cfg, dict) or cfg.get("refinement") or cfg.get("overnight"):
        raise ValueError("Expected a resolved native acquisition configuration")
    training, task = cfg.get("training", {}), cfg.get("task", {})
    worlds = training.get("world_count")
    if type(worlds) is not int or worlds <= 0 or cfg.get("scene", {}).get("world_count") != worlds:
        raise ValueError("Native scene/training world counts must be positive and equal")
    for name, value in (("replay capacity", training.get("replay", {}).get("capacity")),
                        ("replay batch", training.get("replay", {}).get("batch_size")),
                        ("optimizer horizon", training.get("optimization", {}).get("learning_rate_schedule_updates")),
                        ("native target", training.get("total_environment_steps"))):
        if type(value) is not int or value <= 0:
            raise ValueError(f"Invalid {name}")
    if task.get("source_frame_playback") is not True or task.get("motion_start_frame") != 0 or task.get("motion_end_frame") != -1:
        raise ValueError("Native pipeline requires the full source-frame sequence")
    parquet = task.get("parquet")
    if not isinstance(parquet, str) or not Path(parquet).is_absolute() or not Path(parquet).is_file():
        raise ValueError("Native task parquet must be an existing absolute path")
    if cfg.get("env", {}).get("objective", {}).get("_target_") != "flash_chord.objectives.recon_body.ReconBodyObjectiveConfig":
        raise ValueError("Native acquisition requires the ReconBody objective")
    curriculum = cfg.get("curriculum")
    if not isinstance(curriculum, dict) or training.get("curriculum") != curriculum:
        raise ValueError("Native curriculum must be resolved and identical in training")
    thresholds, stages = curriculum.get("thresholds", []), curriculum.get("stages", [])
    if (not thresholds or len(thresholds) != len(stages)
            or any(type(x) is not int or x <= 0 for x in thresholds)
            or any(a >= b for a, b in zip(thresholds, thresholds[1:]))):
        raise ValueError("Native curriculum needs ordered transition thresholds and matching stages")
    return worlds


def build_fresh_config(saved, *, output, end, checkpoint_interval):
    """Fresh native initialization with configured method/curriculum preserved."""
    cfg = normalize_checkpoint_config(saved)
    worlds = validate_native_config(cfg)
    training = cfg["training"]
    if "reference" in cfg or training.get("resume") is not False or training.get("checkpoint_path") is not None:
        raise ValueError("Fresh native config requires no checkpoint/reference state")
    if training.get("checkpoint", {}).get("load_replay_path") is not None:
        raise ValueError("Fresh native config cannot restore replay")
    if end <= 0 or end % worlds or end > training["total_environment_steps"]:
        raise ValueError("Fresh target must align with worlds and stay within its configured target")
    training.update(total_environment_steps=end, resume=False, checkpoint_path=None)
    training["checkpoint"].update(save_interval_environment_steps=checkpoint_interval,
                                  save_replay=False, load_replay_path=None,
                                  load_optimizer=False, load_reward_normalizer=False)
    cfg["logging"].update(mode="offline", log_interval=25,
                          upload_checkpoints=False, upload_training_state=False)
    cfg["output_dir"] = str(output / "checkpoints")
    return cfg


def build_native_resume_config(saved, *, state, output, start, end, checkpoint_interval):
    """Restore saved native curriculum/optimizer clocks with fresh replay/simulator."""
    cfg = copy.deepcopy(saved)
    worlds = validate_native_config(cfg)
    if not 0 < start < end <= cfg["training"]["total_environment_steps"] or start % worlds or end % worlds:
        raise ValueError("Native resume requires a partial aligned checkpoint and its saved endpoint")
    training = cfg["training"]
    training.update(total_environment_steps=end, resume=True, checkpoint_path=str(state))
    training["checkpoint"].update(save_interval_environment_steps=checkpoint_interval,
                                  save_replay=False, load_replay_path=None,
                                  load_optimizer=True, load_reward_normalizer=True)
    cfg["output_dir"] = str(output / "checkpoints")
    return cfg
