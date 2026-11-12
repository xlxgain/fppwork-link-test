import json
import numpy as np
import pytest
from e01 import layout,TAU,wrap,estimate
from e01_c import geometry,support,ROOT

@pytest.mark.parametrize('orientation',['flat','pointy'])
def test_local_support(orientation):
    l,c,inc,edges=layout(orientation=orientation)
    y,x=np.indices(l.shape); xy=np.c_[x.ravel(),y.ravel()]
    es=support(l,c,edges,np.ones(l.shape,bool),inc)
    assert any(e['connection']=='internal' for e in es)
    assert any(e['connection']=='truncated' for e in es)
    for e in es:
        selected=np.unique(np.r_[e['ix'],e['pairs'].ravel()])
        along=(xy[selected]-e['mid'])@e['tangent']
        assert np.all(along>=e['lo']-1e-10) and np.all(along<=e['hi']+1e-10)
        assert np.all(np.isin(l.ravel()[selected],[e['i'],e['j']]))
        assert np.all(np.isin(e['pairs'],e['ix']))
        if e['valid']:
            truth=TAU*x/20+.0003*x*x+.0002*x*y
            m=np.arange(len(c))%9
            for _,mu,_,_,valid,_ in estimate(truth-TAU*m[l],wrap(truth),e,1):
                if valid: assert abs(mu-(m[e['i']]-m[e['j']]))<1e-10

def test_geometry():
    p=json.loads((ROOT/'parameters_C.json').read_text())
    phi,mask,z,u,v=geometry(p)
    y,x=np.indices(z.shape); X=z*(x-p['cx'])/p['fc']; Y=z*(y-p['cy'])/p['fc']
    assert np.max(abs(z-(800+.1*X+.05*Y)))<1e-10
    assert np.max(abs((u-p['pcx'])/p['fp']*z+100-X))<1e-10
    assert np.max(abs((v-p['pcy'])/p['fp']*z-Y))<1e-10
    assert np.allclose(phi[mask],TAU*u[mask]/40)
    _,_,zb,_,_=geometry(p,(0,0))
    assert np.max(abs(zb-(800+.1*zb*(x-127.5)/400+.05*zb*(y-127.5)/400-12*np.exp(-((zb*(x-127.5)/400)**2+(zb*(y-127.5)/400)**2)/(2*25**2)))))<1e-9
    p['baseline_mm']=10000
    phi,mask,*_=geometry(p)
    assert not mask.any() and np.isnan(phi).all()
