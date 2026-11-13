import json
import numpy as np
import pytest
from baselines import h2002,hierarchical,TAU,wrap,hex_outputs
from observations import prepare,generate,ROOT,BASE
from eval_e06 import evaluate

@pytest.mark.parametrize('quadratic',[False,True])
def test_h2002_masks_and_reproducibility(quadratic):
    y,x=np.indices((60,80)); truth=.4*x+.04*y+(0.0005*x*y if quadratic else 0)
    phi=wrap(truth); original=phi.copy(); mask=((x-30)**2+(y-30)**2<24**2); mask[:,38:41]=False
    a,comp,d=h2002(phi,mask); changed=phi.copy(); changed[~mask]=99999; b,_,_=h2002(changed,mask)
    assert np.array_equal(phi,original) and np.array_equal(a[mask],b[mask]) and np.isnan(a[~mask]).all()
    for c in np.unique(comp[mask]):
        region=comp==c; e=(a[region]-truth[region])/TAU; assert np.ptp(e)<1e-12
    assert np.max(abs(wrap(a[mask]-phi[mask])))<1e-12

@pytest.mark.parametrize('scene',['plane','smooth'])
def test_initialization_mechanism_and_E05_input(scene):
    cfg=json.loads((ROOT/'config.json').read_text()); z,u,mask,p,center=prepare(scene,cfg)
    old=json.loads((BASE/'E05_frame_equivalence/outputs/input_manifest.json').read_text())
    record=next(v for v in old['paired_simulation_inputs'] if v['scene']==scene and v['sigma']==5 and v['repeat']==0)
    for minimum in cfg['a_min']:
        phases,zones,a,info=generate(u,z,p,center,scene,minimum,0,0,cfg)
        assert np.max(abs(wrap(phases[60]-TAU*60*u/1024)))<1e-12
        for name,freq in cfg['protocols'].items():
            out=hierarchical({f:phases[f] for f in freq},freq)
            assert np.max(abs(out[mask]-TAU*60*u[mask]/1024))<1e-10
    phases,_,_,info=generate(u,z,p,center,scene,1,5,0,cfg)
    assert next(v['input_hash'] for v in info if v['frequency']==60)==record['frame_hashes']['60']
    assert list(cfg['protocols'].values())==[[1,60],[1,8,60],[1,4,15,60],[1,3,8,22,60]]

def test_global_roi_evaluator_partial_and_gauge():
    truth=np.zeros((2,5)); mask=np.ones(truth.shape,bool); zones=np.array([[0,0,0,1,1],[1,2,2,2,2]])
    psi=truth+3*TAU; psi.ravel()[6:]=np.nan; psi.ravel()[0]+=TAU
    s,roi,a,r,aligned=evaluate(psi,truth,truth,mask,zones)
    assert s['coverage_common']==.6 and s['covered_error']==1/6 and s['relative_integer_error_total']==.5
    assert sum(v['N_all'] for v in roi if v['roi'] in ['core','transition','outside'])==10
    assert sum(v['N_bad_out'] for v in roi if v['roi'] in ['core','transition','outside'])==1
    # All one input component; different algorithm pieces cannot align separately.
    psi=truth.copy(); psi[:,2:]+=TAU
    s,*_=evaluate(psi,truth,truth,mask,zones); assert s['relative_integer_error_total']>0

def test_algorithm_regression_and_config():
    cfg=json.loads((ROOT/'config.json').read_text()); z,u,mask,p,center=prepare('plane',cfg)
    phases,zones,a,info=generate(u,z,p,center,'plane',1,0,0,cfg)
    saved=np.load(BASE/'E05_frame_equivalence/outputs/maps/plane_sigma0_repeat00.npz')
    for name,phase,comp,d,e in hex_outputs(phases[60],mask,'flat'):
        k=saved['integer_orders'][saved['methods'].tolist().index(name)]; finite=np.isfinite(phase)
        difference=np.rint((phase[finite]-phases[60][finite])/TAU)-k[finite]
        assert len(np.unique(difference))==1
    assert cfg['side_px']==30 and cfg['support_halfwidth_px']==6 and cfg['endpoint_margin_px']==6 and cfg['rho_count']==40 and cfg['H2002_rng']==0
