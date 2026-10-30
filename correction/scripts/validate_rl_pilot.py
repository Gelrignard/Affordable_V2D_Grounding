"""Validate a paired export; the CLI writes validation.json, not a physics verdict."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def validate(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    expected={(tier,episode) for tier in manifest['selection']['tiers'] for episode in manifest['selection']['episodes']}
    actual={(row['tier'],row['episode_index']) for row in manifest['entries']}
    if actual!=expected or len(actual)!=len(manifest['entries']):raise ValueError('Missing or duplicate tier/episode entries')
    rows=[]
    for entry in manifest['entries']:
        folder=root/entry['path'];mfile=folder/'manifest.json'
        if hashlib.sha256(mfile.read_bytes()).hexdigest()!=entry['manifest_sha256']:raise ValueError('Episode manifest hash mismatch')
        record=json.loads(mfile.read_text());n=record['frames']
        for file,info in record['files'].items():
            if hashlib.sha256((folder/file).read_bytes()).hexdigest()!=info['sha256']:raise ValueError(f'File hash mismatch: {folder/file}')
        for name in ('original_reference.npz','corrected_reference.npz'):
            file=folder/name
            if not file.exists():
                if name=='corrected_reference.npz' and not entry['corrected_reference_available']:continue
                raise ValueError(f'Missing {file}')
            with np.load(file,allow_pickle=False) as a:
                assert len(a['timestamps'])==n
                np.testing.assert_array_equal(a['source_frame_indices'],np.arange(n))
                assert np.all(np.diff(a['timestamps'])>0)
                assert a['joint_positions'].shape==(n,len(a['joint_names']))
                for key in ('root_position','joint_positions','joint_velocity','ee_pose_w','object_position','root_wxyz','object_wxyz'):
                    assert np.isfinite(a[key]).all(),(file,key)
                for key in ('root_wxyz','object_wxyz'):
                    np.testing.assert_allclose(np.linalg.norm(a[key],axis=1),1.,atol=1e-7)
                assert a['object_observed_mask'].shape==(n,)
        if entry['corrected_reference_available']:
            with np.load(folder/'contact_guidance.npz',allow_pickle=False) as guidance:
                channels=len(guidance['contact_body_names'])
                assert guidance['contact_active'].shape==(n,channels)
                for key in ('force_ff','robot_local_points','object_local_points','inward_normals_local'):
                    assert guidance[key].shape==(n,channels,3),(folder,key)
                    assert np.isfinite(guidance[key]).all(),(folder,key)
                assert len(guidance['contact_geom_names'])==channels
            with np.load(folder/'original_reference.npz',allow_pickle=False) as a,np.load(folder/'corrected_reference.npz',allow_pickle=False) as b:
                for key in ('timestamps','source_frame_indices','joint_names','object_position','object_wxyz','object_observed_mask'):
                    np.testing.assert_array_equal(a[key],b[key])
        files=[folder/name for name in ('original.npz','mechanics.npz')]
        if all(p.exists() for p in files):
            with np.load(files[0],allow_pickle=False) as a,np.load(files[1],allow_pickle=False) as b:
                for result in (a,b):
                    assert len(result['q'])==n and np.isfinite(result['q']).all()
                    assert len(result['object_positions'])==n and np.isfinite(result['object_positions']).all()
                    assert np.all(np.diff(result['timestamps'])>0)
                    assert float(result['max_object_applied_force'])==0.
                np.testing.assert_array_equal(a['timestamps'],b['timestamps'])
                for key in ('q','object_positions','object_wxyz','joint_velocities'):
                    np.testing.assert_array_equal(a[key][0],b[key][0])
        rows.append(dict(tier=entry['tier'],episode_index=entry['episode_index'],frames=n,
            corrected_reference_available=entry['corrected_reference_available'],physical_accepted=entry['physical_accepted']))
    for episode in manifest['selection']['episodes']:
        pair=[json.loads((root/e['path']/'manifest.json').read_text()) for e in manifest['entries'] if e['episode_index']==episode]
        assert len({p['source']['source_sha256'] for p in pair})==len(pair),'Clean/noisy source file unexpectedly identical'
    return dict(schema='v2d_rl_pilot_validation_v1',passed=True,scope='Hashes, arrays, provenance and complete paired recording; does not certify manipulation',entries=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();result=validate(args.root)
    (args.root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
