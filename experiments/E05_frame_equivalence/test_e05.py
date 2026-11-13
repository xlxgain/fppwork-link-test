import numpy as np
from algorithms import integer_sync,hierarchical,TAU,wrap,hex_outputs
from evaluation_e05 import evaluate
from run_e05 import simulated_geometry,simulated_observations,ROOT,BASE
import json

def test_sync_sign_and_disconnection():
    k,comp,d=integer_sync(5,[(0,1),(1,2),(0,2)],[2,-3,-1])
    assert np.array_equal(k[:3],[0,-2,1])
    assert len(np.unique(comp))==3 and d['objective']==0
    k,comp,d=integer_sync(3,[(0,1),(1,2),(0,2)],[2,2,7])
    assert d['integrality_error']<1e-7 and np.isclose(d['objective'],3)

def test_temporal_initialization_and_same_high_phase():
    u=np.linspace(.1,.9,100).reshape(10,10); phases={f:wrap(TAU*f*u) for f in [1,2,4,14,60]}
    full=hierarchical(phases,[1,2,4,14,60]); ambiguous=hierarchical(phases,[2,4,14,60])
    assert np.allclose(full,TAU*60*u)
    assert np.any(abs(ambiguous-TAU*60*u)>TAU)
    for output in [full,ambiguous]: assert np.max(abs(wrap(output-phases[60])))<1e-12

def test_evaluation_gauges_and_single_global_anchor():
    y,x=np.indices((8,10)); truth=.3*x; phi=wrap(truth); mask=np.ones(truth.shape,bool); mask[:,4:6]=False
    psi=truth+3*TAU; psi[:,6:]+=TAU
    s,a,b=evaluate(psi,phi,truth,mask,0)
    assert s['relative_correct_all_common']==1 and s['absolute_correct_all_common']==0
    assert s['anchored_correct_all_common']==.5

def test_whole_field_and_graph_equivalence_noiseless():
    y,x=np.indices((96,96)); truth=.31*x+.02*y; phi=wrap(truth); mask=np.ones(truth.shape,bool)
    for name,result,comp,diag,edges in hex_outputs(phi,mask,'flat'):
        score,_,_=evaluate(result,phi,truth,mask,0)
        assert score['relative_correct_all_common']==1 and score['wrap_closure']<1e-12

def test_geometry_and_pairing():
    p=json.loads((BASE/'E01_hex_pairwise/parameters_C.json').read_text()); cfg=json.loads((ROOT/'config.json').read_text())
    z,u,mask=simulated_geometry('depth_step',p,cfg['step_mm']); y,x=np.indices(z.shape)
    X=z*(x-p['cx'])/p['fc']; Y=z*(y-p['cy'])/p['fc']
    assert np.max(abs(z-(800+.1*X+.05*Y-40*(X>=0))))<1e-10
    a,h=simulated_observations(u,cfg,2,5,0); b,k=simulated_observations(u,cfg,2,5,0)
    assert h==k and np.array_equal(a[60],b[60]) and 1 in a
