"""Full-timeline staged pilot correction; failed stages stay explicitly labeled.

Uses same-tier observations only. Repeated hand modes are inferred before
replay. This extends the staged runner, not the optimization to joint MPC.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
from correction.src.preprocessing import retarget_bridge
from v2d_retarget.core.robot import G1Kinematics
from correction.src.preprocessing.strategy_bundle import read_npz
from correction.src.preprocessing.convex_object import decompose
from correction.src.preprocessing.supported_fixture import regularize_fixture
from correction.src.preprocessing.pilot_phases import infer_phases
from correction.src.preprocessing.interaction_evidence import regions
from correction.src.dynamics.supported_robot import SupportedRobotScene
from correction.src.dynamics.contact_controller import ContactControllerConfig
from correction.src.dynamics.staged_observation import StagedObservationConfig
from correction.src.optimization.surface_hypotheses import SurfaceLibrary
from correction.src.optimization.patch_geometry import solve_patch_geometry
from correction.src.optimization.strategy_screen import static_hold_screen
from correction.src.dynamics.execution_contract import ContactExecutionContract
from correction.src.optimization.surface_reference import PhaseTiming,build_reference,smoothstep
from correction.src.optimization.manipulation_sequence import ContactPhase,union_channels
from correction.src.optimization.actuator_force_profile import allocate_profile
from correction.src.optimization.command_smoothing import refine_commands
from correction.src.dynamics.replay_assessment import assessed
from correction.src.pass_traj.experiment_record import snapshot_sources,process_cost


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray)
                              else x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    started=perf_counter();cpu=process_cost()['cpu_seconds']
    source=args.input/'retargeted/contact_aware.npz'
    raw=read_npz(source);meta=json.loads(str(raw['metadata_json']));source_meta=meta['source']
    times=raw['timestamps'];count=len(times)
    report=dict(status='loading',tier=source_meta['tier'],episode=source_meta['episode_index'],
        sequence_id=meta['sequence_id'],frames=count,duration_s=float(times[-1]-times[0]),
        input_directory=str(args.input.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Automatic input-only multi-phase staged pilot; supported pelvis; experimental',
        ready_for_training=False,accepted=False,object_assistance=False,resets_during_episode=0,cases={},
        limitations=['Source-derived default object inertia; local simulator is not official runtime',
            'Inferred phases and handover transitions are hypotheses, not contact labels',
            'Rejected finite candidates are retained for diagnostics; original fallback is explicit'],
        configuration=vars(args),phase_results=[],provenance=snapshot_sources(args.output,[Path(__file__),
            *[Path('correction/src')/p for p in ['preprocessing/pilot_phases.py','optimization/surface_hypotheses.py',
            'optimization/rigid_grasp.py','optimization/actuator_force_profile.py','optimization/command_smoothing.py',
            'dynamics/staged_observation.py','optimization/observed_force_update.py']]]))
    def save():
        report.update(wall_seconds=perf_counter()-started,cost=process_cost())
        report['cost']['cpu_seconds']-=cpu
        write_json(args.output/'report.json',report)
    save()
    try:
        q,fixture=regularize_fixture(raw['q'],21);raw={**raw,'q':q}
        mesh=trimesh.load(str(raw['object_mesh']),force='mesh')
        parts=decompose(str(raw['object_mesh']),args.input/'collision')
        vertices=np.concatenate([trimesh.load(p,force='mesh').vertices for p,_ in parts])
        bottom=float((Rotation.from_quat(raw['object_wxyz'][0],scalar_first=True).apply(vertices)+raw['object_positions'][0])[:,2].min())
        floor=bottom if abs(bottom)<=.03 else 0.
        physics=dict(mass_kg=1.,friction=1.,timestep_s=.001,
                     object_inertial=dict(com_local_m=[0.,0.,0.],diagonal_kg_m2=[.01]*3))
        scene=SupportedRobotScene(G1Kinematics(),mesh.bounds,source_meta['support_surface'],q[0],
            mass=1.,friction=1.,dt=.001,move_waist=True,moving_fixture=True,object_parts=parts,
            floor_z=floor,object_inertial=physics['object_inertial'])
        (args.output/'scene.xml').write_text(scene.xml,encoding='utf-8')
        library=SurfaceLibrary(scene);contract=ContactExecutionContract(friction=.8,physics_friction=1.)
        phase_specs,phase_audit=infer_phases(raw['contact_active'],times)
        report.update(status='screening',physics=physics,fixture_regularization=fixture,
            floor_z_m=floor,initial_object_bottom_m=bottom,phase_inference=phase_audit,phase_specs=phase_specs)
        save()
        phases=[];commands=[];composed=q.copy();covered=np.zeros(count,bool)
        for phase_id,spec in enumerate(phase_specs):
            phase_start=perf_counter();timing=PhaseTiming(**spec['timing']);fits=[];candidates=[]
            budget=max(4.,args.screen_seconds/max(1,len(phase_specs)))
            for frame in spec['frames']:
                normal={r.side:np.asarray(r.outward_object) for r in regions(raw,max(0,frame-10),min(count-1,frame+10))}
                for family in ('observed_six','coherent_four','support_any'):
                    if perf_counter()-phase_start>=budget:break
                    row=dict(frame=frame,family=family)
                    try:
                        points=library.hypothesis(raw,frame,family,normal,sides=spec['sides'])
                        grasp,points,geometry=solve_patch_geometry(scene,points,q[frame],raw['object_positions'][frame],
                            raw['object_wxyz'][frame],seconds=min(5.,max(.1,budget-(perf_counter()-phase_start))))
                        force,gap=static_hold_screen(scene,library,points,floor,contract)
                        valid=bool(geometry['max_gap_m']<=.003 and geometry['max_robot_penetration_m']<=.002 and
                            geometry['max_normal_cone_gap_m']<=.001 and geometry['patch_neighborhood_satisfied'] and
                            geometry['evidence_region_satisfied'])
                        row.update(geometry=geometry,force_screen=force,geometry_accepted=valid,
                                   accepted_candidate=bool(valid and force['feasible']))
                        score=(not row['accepted_candidate'],not valid,geometry['max_gap_m']+5*geometry['max_robot_penetration_m'])
                        fits.append((score,grasp,points,row))
                    except (ValueError,KeyError) as e:row['error']=str(e)
                    candidates.append(row)
            audit=dict(phase_id=phase_id,spec=spec,candidates=candidates,corrected=False)
            report['phase_results'].append(audit)
            if not fits:
                audit['failure']='No finite candidate; original motion retained for this mode';save();continue
            _,grasp,points,chosen=min(fits,key=lambda row:row[0])
            try:
                force=np.asarray(chosen['force_screen']['forces'])[:len(points)]
                command,construction=build_reference(scene,raw,points,grasp,force,chosen['frame'],
                    approach=True,rigid_grasp=True,release=True,phase=timing,continuous_ik=True)
                begin,end=timing.approach_start,timing.hold_end
                ids=np.arange(begin,end+1)
                blend=smoothstep((ids-begin)/max(1,timing.contact_start-begin))*smoothstep((end-ids)/max(1,end-timing.contact_end))
                cols=7+scene.movable
                composed[np.ix_(ids,cols)]=(1-blend[:,None])*q[np.ix_(ids,cols)]+blend[:,None]*command['q'][np.ix_(ids,cols)]
                covered[ids[blend>0]]=True
                phases.append(ContactPhase(points,grasp,chosen['frame'],force,timing));commands.append(command)
                audit.update(corrected=True,selected=chosen,construction=construction,contacts=[asdict(p) for p in points])
            except (ValueError,RuntimeError) as e:
                audit['failure']=f'Construction failed: {e}; original motion retained'
            audit['wall_seconds']=perf_counter()-phase_start;save()
            print('phase',phase_id,'corrected',audit['corrected'],flush=True)
        if not phases:
            report.update(status='no_contact_candidate',diagnostic_fallback='Original retargeting only; no fabricated corrected trajectory')
            np.savez_compressed(args.output/'baseline_only.npz',**raw)
            return
        # Force stages use the composed nominal motion; each phase keeps its
        # own physical surface identities. All physical replays remain whole.
        for phase,command in zip(phases,commands):
            audit=next(a for a in report['phase_results'] if a.get('selected',{}).get('frame')==phase.frame and a['corrected'])
            sl=slice(phase.timing.approach_start,phase.timing.hold_end+1)
            blend=smoothstep((np.arange(count)-phase.timing.contact_start)/(phase.timing.force_full-phase.timing.contact_start))
            blend*=smoothstep((phase.timing.force_end-np.arange(count))/(phase.timing.force_end-phase.timing.contact_end))
            try:
                force,profile=allocate_profile(scene,phase.points,composed[sl],raw['object_positions'][sl],raw['object_wxyz'][sl],
                    times[sl],blend[sl],friction=.8,include_environment=True)
                profile.pop('environment_plan',None)
                command['force_ff'][:]=0.;command['force_ff'][sl]=force;audit['force_profile']=profile
            except ValueError as e:
                command['force_ff'][:]=0.;audit['force_profile']=dict(solved=False,error=str(e),fallback='Zero force; geometry-only diagnostic')
        points,fields=union_channels(phases,commands,count)
        reference=dict(q=composed,command_velocity=PchipInterpolator(times,composed[:,7+scene.movable],axis=0)(times,1),
            timestamps=times,**fields,baseline_command_q=q,initial_q=q[0].copy(),initial_velocity=np.zeros(len(scene.movable)),
            object_positions=raw['object_positions'],object_wxyz=raw['object_wxyz'],object_mesh=raw['object_mesh'],
            bounds=mesh.bounds,floor_z=floor,frame_ids=np.asarray(source_meta['selected_source_indices']),
            object_observed_mask=np.asarray(source_meta['object_observed_mask']),correction_coverage_mask=covered,
            joint_names=np.asarray(meta['joint_names']),movable_joint_indices=scene.movable)
        np.savez_compressed(args.output/'before_spline.npz',**reference)
        report['status']='spline_refinement';save()
        fitted,audit=refine_commands(scene,reference,points,seconds=args.spline_seconds,free_trust_rad=.12,continuous_bounds=True)
        report['optimization']=audit
        if fitted is not None:reference=fitted
        else:report['diagnostic_fallback']='Spline rejected; finite composed pre-spline trajectory retained'
        report['planning_accepted']=bool(fitted is not None and not phase_audit['rejected_intervals'] and
            len(phases)==len(phase_specs) and all(p.get('selected',{}).get('accepted_candidate',False) and
                p.get('force_profile',{}).get('task_wrench_satisfied',False) for p in report['phase_results']))
        report.update(contacts=[asdict(p) for p in points],phase_contacts=[dict(frame=p.frame,timing=asdict(p.timing),
            contacts=[asdict(x) for x in p.points]) for p in phases],corrected_frames=int(covered.sum()))
        np.savez_compressed(args.output/'reference.npz',**reference)
        np.savez_compressed(args.output/'planning_poses.npz',object_positions=raw['object_positions'],object_wxyz=raw['object_wxyz'])
        config=ContactControllerConfig(interface='torque',feedback_gain=.25,friction=.8,patch_distance_gate=True,near_surface_m=.025)
        report.update(controller=asdict(config),kp_hand=2.,inertia_ff_scale=1.)
        for case in ('original','mechanics'):
            report['status']='replaying_'+case;save();start=perf_counter()
            extra={} if case=='original' or args.feedback_gain==0 else dict(staged_observation=StagedObservationConfig(gain=args.feedback_gain),
                staged_observation_plan=dict(object_positions=raw['object_positions'],object_wxyz=raw['object_wxyz']))
            result=scene.rollout(q if case=='original' else reference['q'],raw['object_positions'],raw['object_wxyz'],times,
                force_ff=np.zeros_like(reference['force_ff']) if case=='original' else reference['force_ff'],force_points=points,
                contact_controller=config,contact_normals_local=reference['inward_normals_local'],
                contact_active=np.zeros_like(reference['contact_active']) if case=='original' else reference['contact_active'],
                point_local_trajectory=reference['robot_local_points'],object_local_trajectory=reference['object_local_points'],
                command_velocity=None if case=='original' else reference['command_velocity'],kp_arm=300.,kp_hand=2.,
                inertia_ff=True,inertia_ff_scale=1.,smooth_fixture=True,initial_q=reference['initial_q'],
                initial_velocity=reference['initial_velocity'],contact_body_audit=True,**extra)
            if len(result['q'])!=count or result['max_object_applied_force']!=0:raise AssertionError('Expected complete unassisted episode')
            np.savez_compressed(args.output/f'{case}.npz',**result)
            report['cases'][case]=dict(**assessed(result,times),wall_seconds=perf_counter()-start)
            if extra:
                report['observation_audit']={k.removeprefix('staged_observation_'):v.item() if isinstance(v,np.ndarray) else v
                    for k,v in result.items() if k.startswith('staged_observation_')}
            save();print(case,report['cases'][case]['rms_position_error_mm'],flush=True)
        feedback_ok=report.get('observation_audit',{}).get('rate_conflicts',0)==0 and report.get('observation_audit',{}).get('failures',0)==0
        report.update(status='complete',accepted=bool(report['planning_accepted'] and report['cases']['mechanics']['accepted_physics'] and feedback_ok))
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--screen-seconds',type=float,default=75.)
    parser.add_argument('--spline-seconds',type=float,default=30.)
    parser.add_argument('--feedback-gain',type=float,default=1.)
    run(parser.parse_args())
