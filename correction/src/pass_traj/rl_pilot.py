"""Package paired references and physical evidence, preserving failure labels."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from correction.src.preprocessing import retarget_bridge
from correction.src.preprocessing.strategy_bundle import read_npz
from v2d_retarget.core.robot import G1Kinematics


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_arrays(q,times,raw,kin,source_indices):
    names=['left_hand_palm_link','right_hand_palm_link']
    poses=np.asarray([[np.r_[p.translation,Rotation.from_matrix(p.rotation).as_quat(scalar_first=True)]
                       for p in kin.poses(row,names)] for row in q])
    source=json.loads(str(raw['metadata_json']))['source']
    return dict(timestamps=times,source_frame_indices=np.asarray(source_indices),
        joint_names=np.asarray(kin.joint_names),root_position=q[:,:3],root_wxyz=q[:,[6,3,4,5]],
        joint_positions=q[:,7:],joint_velocity=CubicSpline(times,q[:,7:],axis=0)(times,1),
        ee_link_names=np.asarray(names),ee_pose_w=poses,object_position=raw['object_positions'],object_wxyz=raw['object_wxyz'],
        object_observed_mask=np.asarray(source['object_observed_mask']),world_from_source_cv=np.asarray(source['world_from_source_cv']),
        coordinate_frame='world_z_up',quaternion_order='wxyz',units='metres_seconds_radians',
        velocity_semantics='Derivative of cubic joint reference interpolant, matching baseline replay; not measured robot velocity')


def export_pilot(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((root/'selection.json').read_text())
    kin=G1Kinematics();entries=[]
    for ep in config['episodes']:
        for tier in config['tiers']:
            src=root/f'tier{tier}'/f'episode_{ep:06d}';out=output/f'tier{tier}'/f'episode_{ep:06d}'
            out.mkdir(parents=True);assets=out/'assets';assets.mkdir()
            raw=read_npz(src/'retargeted/contact_aware.npz');meta=json.loads(str(raw['metadata_json']));smeta=meta['source']
            if smeta['tier']!=tier or smeta['episode_index']!=ep:raise ValueError('Tier/episode provenance mismatch')
            indices=smeta['selected_source_indices'];n=len(raw['timestamps'])
            if indices!=list(range(n)):raise ValueError('Full original timeline required')
            report=json.loads((src/'correction/report.json').read_text())
            item=dict(tier=tier,episode_index=ep,sequence_id=meta['sequence_id'],frames=n,
                duration_s=float(raw['timestamps'][-1]-raw['timestamps'][0]),source=smeta,
                source_retarget_sha256=digest(src/'retargeted/contact_aware.npz'),
                accepted=bool(report.get('accepted',False)),
                physical_accepted=bool(report.get('cases',{}).get('mechanics',{}).get('accepted_physics',False)),
                ready_for_training=False,
                status=report['status'],planning_accepted=report.get('planning_accepted',False),
                scope='Experimental references, not certified expert demonstrations or learned policies',
                local_tracking_reference='Tier-specific input object poses, not clean Tier 1 evaluation targets',
                correction_coverage_frames=report.get('corrected_frames',0),metrics=report.get('cases',{}),
                quaternion_order='wxyz',joint_order=kin.joint_names,physics=report.get('physics'),files={})
            baseline_q=raw['q']
            ref_path=src/'correction/reference.npz'
            reference=read_npz(ref_path) if ref_path.exists() else None
            if reference is not None:baseline_q=reference['baseline_command_q']
            np.savez_compressed(out/'original_reference.npz',**reference_arrays(baseline_q,raw['timestamps'],raw,kin,indices))
            if reference is not None:
                if len(reference['q'])!=n or not np.isfinite(reference['q']).all():raise ValueError('Invalid corrected reference')
                corrected=reference_arrays(reference['q'],raw['timestamps'],raw,kin,indices)
                # Active controller velocities are saved by the actual command
                # interpolant. Do not overwrite them with generic finite differences.
                corrected['joint_velocity'][:,reference['movable_joint_indices']]=reference['command_velocity']
                corrected['velocity_semantics']='Saved active command velocities; other joint derivatives from reference interpolant'
                corrected['correction_coverage_mask']=reference['correction_coverage_mask']
                np.savez_compressed(out/'corrected_reference.npz',**corrected)
                force_fields={k:reference[k] for k in ('timestamps','force_ff','contact_active','robot_local_points',
                    'object_local_points','inward_normals_local','movable_joint_indices','initial_q','initial_velocity')}
                model=mujoco.MjModel.from_xml_path(str((src/'correction/scene.xml').resolve()))
                force_fields['contact_body_names']=np.asarray([model.body(p['body']).name for p in report['contacts']])
                force_fields['contact_geom_names']=np.asarray([model.geom(p['geom']).name for p in report['contacts']])
                static_mask=np.zeros(n,bool);wrench_mask=np.zeros(n,bool)
                for phase in report['phase_results']:
                    timing=phase['spec']['timing'];sl=slice(timing['contact_start'],timing['force_end']+1)
                    static_mask[sl]=phase.get('selected',{}).get('accepted_candidate',False)
                    wrench_mask[sl]=phase.get('force_profile',{}).get('task_wrench_satisfied',False)
                force_fields.update(phase_has_accepted_static_seed=static_mask,
                    phase_has_accepted_nominal_wrench=wrench_mask,
                    command_spline_accepted=bool(report.get('optimization',{}).get('solved',False)),
                    physical_episode_accepted=bool(item['physical_accepted']))
                np.savez_compressed(out/'contact_guidance.npz',**force_fields)
                shutil.copy2(ref_path,out/'native_controller_reference.npz')
                item['corrected_reference_available']=True
                item['reference_kind']='optimized_spline' if report.get('optimization',{}).get('solved',False) and not report.get('diagnostic_fallback') else 'diagnostic_pre_spline_candidate'
            else:
                item.update(corrected_reference_available=False,reference_kind='original_only_no_correction_available')
            for name in ('original.npz','mechanics.npz','planning_poses.npz'):
                source=src/'correction'/name
                if source.exists():shutil.copy2(source,out/name)
            shutil.copy2(src/'correction/report.json',out/'correction_report.json')
            if (src/'prepare_report.json').is_file():
                shutil.copy2(src/'prepare_report.json',out/'preparation_report.json')
            shutil.copy2(src/'human_motion.npz',out/'human_motion_source.npz')
            shutil.copy2(str(raw['object_mesh']),assets/'object.glb')
            shutil.copy2(smeta['support_surface'],assets/'support.usda')
            if (src/'collision').exists():shutil.copytree(src/'collision',assets/'collision')
            item['asset_paths']=dict(object_mesh='assets/object.glb',support_surface='assets/support.usda',
                robot='External G1/Dex3 dependency; verify joint order and URDF hash',urdf_sha256=meta['urdf_sha256'])
            item['native_path_notice']='Original source/report/native NPZ paths are preserved provenance. Use asset_paths to relocate; normalized reference arrays contain no filesystem dependencies.'
            for file in sorted(out.rglob('*')):
                if file.is_file():item['files'][file.relative_to(out).as_posix()]=dict(sha256=digest(file),bytes=file.stat().st_size)
            (out/'manifest.json').write_text(json.dumps(item,indent=2)+'\n',encoding='utf-8')
            entries.append(dict(tier=tier,episode_index=ep,path=out.relative_to(output).as_posix(),frames=n,
                status=item['status'],reference_kind=item['reference_kind'],corrected_reference_available=item['corrected_reference_available'],
                planning_accepted=item['planning_accepted'],physical_accepted=item['physical_accepted'],
                original_position_rms_mm=item['metrics'].get('original',{}).get('rms_position_error_mm'),
                corrected_position_rms_mm=item['metrics'].get('mechanics',{}).get('rms_position_error_mm'),
                manifest_sha256=digest(out/'manifest.json')))
    manifest=dict(schema='v2d_rl_pilot_v1',selection=config,entries=entries,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        ready_for_training=False,usage='Explicit imperfect-reference RL experiments; inspect rejection reasons and action/controller contract',
        official_evaluation=False,whole_body_validated=False,learned_policy_included=False,
        dataset_source='https://huggingface.co/datasets/nvidia/video_to_data_challenge',
        dataset_license='CC-BY-4.0; preserve source attribution; robot dependencies supplied separately')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    workspace=Path(__file__).resolve().parents[3]
    shutil.copy2(workspace/'docs/correction/PIPELINE.md',output/'METHOD_AND_RL_HANDOFF.md')
    shutil.copy2(workspace/'docs/correction/RL_PILOT_ARTIFACT_FORMAT.md',output/'README.md')
    for name in ('prepared_input_audit.json','batch_summary.json'):
        if (root/name).is_file():shutil.copy2(root/name,output/name)
    print(json.dumps(entries,indent=2),flush=True)
