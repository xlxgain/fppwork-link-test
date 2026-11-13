import numpy as np
import pytest
from run_e04 import phase_maps,temporal_candidate,full_layout,local_support,window_run,wrap,TAU
from e01 import layout,prepare,estimate
from e01_c import support

def test_historical_files_preserved():
    import subprocess
    from run_e04 import ROOT
    subprocess.run(['git','diff','--exit-code','c442a660d1b9e4d2b6b778e29975483bfc13f85d','--','experiments/E01_hex_pairwise','experiments/E02_local_unwrapping','experiments/E03_low_modulation'],cwd=ROOT.parent.parent,check=True,capture_output=True)

def test_four_step_sign_units():
    y,x=np.indices((20,40)); truth=.3*x+.1*y
    images=120+70*np.cos(truth[None]+np.arange(4)[:,None,None]*np.pi/2)
    phi,A,B=phase_maps(images)
    assert np.max(abs(wrap(phi-truth)))<1e-13
    assert np.allclose(A,120) and np.allclose(B,70)
    with pytest.raises(ValueError): phase_maps(images[:3])

@pytest.mark.parametrize('orientation',['flat','pointy'])
def test_geometry_adapter(orientation):
    l,c,inc,edges=layout(orientation=orientation)
    L,C,I,E=full_layout(l.shape,orientation=orientation)
    assert np.array_equal(l,L) and np.allclose(c,C) and np.array_equal(inc,I)
    original=support(l,c,edges,np.ones(l.shape,bool),inc)
    cropped=local_support(L,C,E,np.ones(l.shape,bool),I)
    for a,b in zip(original,cropped):
        assert a['valid']==b['valid'] and np.array_equal(a['ix'],b['ix']) and np.array_equal(a['pairs'],b['pairs'])
        assert np.allclose(a['residual'],b['residual'],atol=1e-12)

def test_temporal_necessary_not_sufficient():
    t=np.linspace(-1,1,100); counts=[2,4,14,60]
    phases=[wrap(t*n/2) for n in counts]
    candidate,_=temporal_candidate(phases,counts)
    assert np.allclose(candidate,t*30)
    # Local wrong order is still congruent. Do not certify by modulo alone.
    wrong=candidate.copy(); wrong[:20]+=TAU
    assert np.max(abs(wrap(wrong-phases[-1])))<1e-12

@pytest.mark.parametrize('method',['W0','W1_MLSQU_HEX_Q2'])
def test_cropped_window_equivalence(method):
    from local_unwrap import unwrap_window
    y,x=np.indices((128,128)); coords=np.stack([x,y],-1)
    labels,c,inc,edges=full_layout(y.shape); mask=np.ones(y.shape,bool)
    phi=wrap(.31*x+.01*y)
    raw,diags=window_run(phi,mask,labels,c,method)
    for i in [len(c)//2]:
        expected,d=unwrap_window(phi,labels==i,coords,dict(method=method,center=c[i]))
        assert np.allclose(raw[labels==i],expected[labels==i],atol=1e-12,equal_nan=True)
