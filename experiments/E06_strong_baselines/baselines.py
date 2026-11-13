"""Four-frame interfaces accept highest-frequency phase and geometry mask only."""
import sys,inspect,time
from pathlib import Path
import numpy as np
from scipy.ndimage import label
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'E05_frame_equivalence'))
from algorithms import hex_outputs,spatial_bfs,hierarchical,TAU,wrap
import skimage
from skimage.restoration import unwrap_phase

def h2002(phi,mask):
    start=time.perf_counter(); comp,n=label(mask); out=np.full(phi.shape,np.nan)
    signature=inspect.signature(unwrap_phase); option='rng' if 'rng' in signature.parameters else 'seed'
    normalized=(np.asarray(phi)+np.pi)%TAU-np.pi
    for c in range(1,n+1):
        y,x=np.nonzero(comp==c); y0,y1=y.min(),y.max()+1; x0,x1=x.min(),x.max()+1
        region=comp[y0:y1,x0:x1]==c
        # Masked values are placeholders, never valid observations.
        arr=np.ma.array(np.where(region,normalized[y0:y1,x0:x1],0.),mask=~region)
        result=unwrap_phase(arr,wrap_around=False,**{option:0})
        a=np.ma.filled(result,np.nan); valid=region&np.isfinite(a)
        if valid.sum()!=region.sum(): raise RuntimeError('H2002 incomplete component')
        original=phi[y0:y1,x0:x1]
        if np.max(abs(wrap(a[valid]-original[valid])))>1e-8: raise RuntimeError('H2002 noncongruent output')
        patch=out[y0:y1,x0:x1]; patch[valid]=original[valid]+TAU*np.rint((a[valid]-original[valid])/TAU)
    return out,comp,dict(runtime=time.perf_counter()-start,library_version=skimage.__version__,signature=str(signature),rng_keyword=option,graph_components=n)

def four_frame(phi,mask):
    for orientation in ['flat','pointy']:
        yield from hex_outputs(phi,mask,orientation)
    a,c,d=spatial_bfs(phi,mask); yield 'global_BFS',a,c,d,[]
    a,c,d=h2002(phi,mask); yield 'H2002_skimage',a,c,d,[]

FOUR=[f'{o}_{w}_{m}' for o in ['flat','pointy'] for w in ['W0','W1_MLSQU_HEX_Q2'] for m in ['M0','M1']]+['global_BFS','H2002_skimage']
