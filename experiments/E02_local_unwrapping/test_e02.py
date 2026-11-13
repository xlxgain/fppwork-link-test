import numpy as np
import pytest
from local_unwrap import unwrap_window,wrap,TAU,penalty
from evaluation import evaluate
from run_e02 import input_images,layout,support,estimate,demodulate

@pytest.mark.parametrize('method',['W0','W1_MLSQU_HEX_Q2'])
@pytest.mark.parametrize('orientation',['flat','pointy'])
def test_hexagonal_mask_isolation(method,orientation):
    labels,c,_,_=layout(orientation=orientation)
    i=int(np.argmin(np.sum((c-128)**2,axis=1)))
    mask=labels==i; y,x=np.indices(labels.shape); coords=np.stack([x,y],-1)
    phi=wrap(.31*x+.0004*x*y)
    a,d=unwrap_window(phi,mask,coords,dict(method=method,center=c[i]))
    phi[~mask]=99999
    b,_=unwrap_window(phi,mask,coords,dict(method=method,center=c[i]))
    assert d['valid'] and np.array_equal(a[mask],b[mask])

@pytest.mark.parametrize('method',['W0','W1_MLSQU_HEX_Q2'])
@pytest.mark.parametrize('quadratic',[False,True])
def test_relative_and_isolation(method,quadratic):
    y,x=np.indices((64,64)); coords=np.stack([x,y],-1)
    mask=((x-32)**2+(y-32)**2)<25**2
    truth=.32*x+.02*y+(0.0005*x*y if quadratic else 0)
    phi=wrap(truth); config=dict(method=method,center=[32,32])
    psi,d=unwrap_window(phi,mask,coords,config)
    s,_=evaluate(psi,phi,truth,mask,d)
    assert s['strict_success'] and s['wrap_consistency_error']<1e-12
    assert len(np.unique(np.rint((psi[mask]-phi[mask])/TAU)))>1
    other=phi.copy(); other[~mask]=np.nan
    p2,_=unwrap_window(other,mask,coords,config)
    assert np.allclose(psi[mask],p2[mask])
    evaluate(psi,phi,truth+TAU*7,mask,d)
    p3,_=unwrap_window(phi,mask,coords,config)
    assert np.array_equal(psi[mask],p3[mask])

def test_evaluator():
    t=np.arange(100).reshape(10,10)*.1; phi=wrap(t); m=np.ones(t.shape,bool)
    d=dict(valid=True,n_components=1)
    s,_=evaluate(t+3*TAU,phi,t,m,d); assert s['strict_success']
    bad=t.copy(); bad[:2,:2]+=TAU
    s,_=evaluate(bad,phi,t,m,d); assert s['relative_integer_error_rate']>0

@pytest.mark.parametrize('method',['W0','W1_MLSQU_HEX_Q2'])
def test_failures(method):
    phi=np.zeros((8,8)); y,x=np.indices(phi.shape); coords=np.stack([x,y],-1)
    m=np.zeros(phi.shape,bool)
    assert not unwrap_window(phi,m,coords,dict(method=method))[1]['valid']
    m[0,0]=m[-1,-1]=True
    assert 'disconnected' in unwrap_window(phi,m,coords,dict(method=method))[1]['reason']
    if method!='W0':
        m[:]=False; m[0,:]=True
        assert not unwrap_window(phi,m,coords,dict(method=method))[1]['valid']

def test_penalty_plain_difference():
    r=np.array([-3.,3.,-3.])
    assert penalty(r,[(np.array([0]),np.array([1])),(np.array([1]),np.array([2]))])==12

def test_shared_input_sign_and_edges():
    truth=np.indices((256,256))[1]*.31
    cfg=dict(seed=2,A=127.5,B=100)
    a,h=input_images(truth,cfg,0,5,0); b,h2=input_images(truth,cfg,0,5,0)
    assert h==h2 and np.array_equal(a,b)
    phi=demodulate(a)
    assert np.max(abs(wrap(demodulate(input_images(truth,cfg,0,0,0)[0])-truth)))<1e-12
    for orientation in ['flat','pointy']:
        l,c,inc,edges=layout(orientation=orientation)
        es=support(l,c,edges,np.ones(l.shape,bool),inc)
        assert len(es)==len(edges) and any(not len(e['pairs']) for e in es)
        e=next(e for e in es if e['valid'])
        psi=truth.copy(); psi[l==e['i']]+=TAU*3; psi[l==e['j']]+=TAU*5
        for _,mu,_,_,valid,_ in estimate(psi,wrap(truth),e,1):
            assert valid and abs(mu-2)<1e-10
