import numpy as np
import pytest
from e01 import wrap, demodulate, layout, prepare, estimate, TAU

def test_wrap_demodulation():
    p=np.linspace(-40,40,200).reshape(10,20)
    images=127.5+100*np.cos(p[None]+np.arange(4)[:,None,None]*np.pi/2)
    assert np.max(np.abs(wrap(demodulate(images)-p)))<1e-13
    assert np.max(np.abs(wrap(wrap(p))-wrap(p)))<1e-14

@pytest.mark.parametrize('orientation',['flat','pointy'])
def test_coverage(orientation):
    labels,centers,incomplete,edges=layout(orientation=orientation)
    assert labels.shape==(256,256) and labels.min()==0
    assert labels.max()+1==len(centers) and incomplete.any() and (~incomplete).any()
    for (i,j),pairs in edges.items():
        assert np.all(labels.ravel()[pairs[:,0]]==i)
        assert np.all(labels.ravel()[pairs[:,1]]==j)
        y1,x1=np.unravel_index(pairs[:,0],labels.shape)
        y2,x2=np.unravel_index(pairs[:,1],labels.shape)
        assert np.all(abs(x1-x2)+abs(y1-y2)==1)

@pytest.mark.parametrize('orientation',['flat','pointy'])
@pytest.mark.parametrize('quadratic',[False,True])
def test_recovery_scale_and_reference(orientation,quadratic):
    labels,centers,_,edges=layout(orientation=orientation)
    base=prepare(labels,centers,edges)
    scaled=prepare(labels,centers,edges,coord_scale=1e-4)
    y,x=np.indices(labels.shape)
    truth=TAU*x/20+.035*y
    if quadratic:
        truth=truth+.0003*x*x+.0002*x*y+.00025*y*y
    m=np.random.default_rng(7).integers(-10,11,len(centers))
    for e,s in zip(base,scaled):
        for shift in [0,17]:
            psi=truth-TAU*(m+shift)[labels]
            for v in [e,s]:
                for method,mu,_,_,valid,_ in estimate(psi,wrap(truth),v,1.):
                    if valid:
                        assert abs(mu-(m[e['i']]-m[e['j']]))<1e-10

def test_reference_changes():
    labels,c,_,edges=layout()
    e=prepare(labels,c,edges)[10]
    y,x=np.indices(labels.shape); truth=TAU*x/20
    m=np.arange(len(c))%7
    n=m.copy(); n[e['i']]+=3; n[e['j']]-=2
    a=estimate(truth-TAU*m[labels],wrap(truth),e,1.)
    b=estimate(truth-TAU*n[labels],wrap(truth),e,1.)
    for u,v in zip(a,b):
        assert abs(v[1]-u[1]-5)<1e-10
