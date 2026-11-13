import json
import numpy as np
import pytest
from run_e03 import E02,ROOT,layout,support,unwrap_window,demodulate,estimate,wrap,TAU,historical_hashes
from radiometry import attenuation,intensities,noise_field,digest
from evaluation_e03 import evaluate,reference,propagation

def test_no_unique_reference_group_and_history():
    import pandas as pd
    import subprocess
    from summarize import pair_summary
    frame=pd.DataFrame([dict(group='empty_reference',both_windows_correct=False,valid=True,integer_error=np.nan,end_to_end_success=False)])
    summary=pair_summary(frame,['group'])
    assert summary.geometric_edges.iloc[0]==1
    assert np.isnan(summary.conditional_integer_error_rate.iloc[0])
    assert summary.end_to_end_pair_success_fraction.iloc[0]==0
    subprocess.run(['git','diff','--exit-code','41ec542b13c70207930c920eb33a8a25baae7824','--','experiments/E01_hex_pairwise','experiments/E02_local_unwrapping'],cwd=ROOT.parent.parent,check=True,capture_output=True)

@pytest.mark.parametrize('scene,index',[('plane',0),('edge',2)])
def test_mechanism_and_regression(scene,index):
    data=np.load(E02/'outputs_C_review'/f'{scene}.npz'); truth=data['phase_radians']
    manifest=json.loads((E02/'outputs_E02/input_manifest.json').read_text()); center=manifest['scenes']['edge']['bump_center_XY_mm']
    cfg=json.loads((ROOT/'config.json').read_text())
    for minimum in cfg['a_min']:
        a,z=attenuation(data['depth_mm'],manifest['geometry'],center,minimum)
        images=intensities(truth,a,np.zeros((4,*truth.shape)))
        assert np.max(abs(wrap(demodulate(images)-truth)))<1e-12
    a=np.ones(truth.shape)
    noise=noise_field(truth.shape,cfg['seed'],index,5,0)
    assert digest(intensities(truth,a,noise))==next(r['input_hash'] for r in manifest['inputs'] if r['scene_id']==scene and r['sigma']==5 and r['repeat']==0)
    example=np.load(E02/'outputs_E02/example_W0.npz') if scene=='edge' else None
    if example is not None:
        phi=demodulate(intensities(truth,a,noise)); assert np.array_equal(phi,example['phi'])
        labels,c,_,_=layout(); y,x=np.indices(truth.shape); coords=np.stack([x,y],-1)
        for method in ['W0','W1_MLSQU_HEX_Q2']:
            old=np.load(E02/'outputs_E02'/f'example_{method}.npz')['psi']
            i=np.argmin(np.sum((c-128)**2,axis=1)); mask=(labels==i)&data['valid']
            output,d=unwrap_window(phi,mask,coords,dict(method=method,center=c[i]))
            assert np.allclose(output[mask],old[mask],rtol=0,atol=1e-12)

def test_partial_missing_and_integer_reference():
    truth=np.zeros((2,5)); phi=truth.copy(); mask=np.ones(truth.shape,bool); zones=np.zeros(truth.shape,int)
    psi=np.zeros(truth.shape); psi.ravel()[6:]=np.nan; psi.ravel()[0]=TAU
    diag=dict(valid=True,n_components=1)
    s,_=evaluate(psi,phi,truth,mask,diag,zones)
    assert s['N_out']==6 and s['N_bad_out']==1 and s['N_missing']==4
    assert s['coverage']==.6 and s['covered_relative_error_rate']==1/6 and s['total_failure_rate']==.5
    empty,_=evaluate(np.full(truth.shape,np.nan),phi,truth,mask,diag,zones)
    assert empty['coverage']==0 and empty['total_failure_rate']==1 and np.isnan(empty['c'])
    good,_=evaluate(truth+3*TAU,phi,truth,mask,diag,zones)
    assert good['strict_success']
    ref=reference(good,s); assert not ref['both_windows_correct'] and np.isnan(ref['d_reference'])

def test_propagation_coefficients():
    labels,c,inc,edges=layout(); es=support(labels,c,edges,np.ones(labels.shape,bool),inc)
    e=next(e for e in es if e['valid']); y,x=np.indices(labels.shape)
    psi=.3*x+.01*y; psi[labels==e['j']]+=TAU*4
    mu=estimate(psi,wrap(psi),e,25/20000)[1][1]
    q=e['residual']/(TAU*e['info'])
    assert abs(q@psi.ravel()[e['ix']]-mu)<1e-12
    v,group=propagation(e,np.full(labels.shape,100.),5)
    J=estimate(psi,wrap(psi),e,25/20000)[1][2]
    assert np.isclose(v,1/J) and group=='higher_snr'
    assert propagation(e,np.full(labels.shape,2.),0)==(0.,'no_noise')

def test_evaluator_isolation():
    y,x=np.indices((20,20)); mask=np.zeros((20,20),bool); mask[2:-2,2:-2]=True
    coords=np.stack([x,y],-1); phi=wrap(.3*x)
    for method in ['W0','W1_MLSQU_HEX_Q2']:
        config=dict(method=method); a,d=unwrap_window(phi,mask,coords,config)
        evaluate(a,phi,phi+17*TAU,mask,d,np.ones(phi.shape,int))
        other=phi.copy(); other[~mask]=9999
        b,_=unwrap_window(other,mask,coords,config)
        assert np.array_equal(a[mask],b[mask])
