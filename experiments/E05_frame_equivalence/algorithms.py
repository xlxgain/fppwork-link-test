"""Observation-only whole-field algorithms. No truth or temporal references."""
import sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack,vstack,eye
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import label
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'E01_hex_pairwise')); sys.path.insert(0,str(BASE/'E02_local_unwrapping')); sys.path.insert(0,str(BASE/'E04_real_monocular'))
from local_unwrap import unwrap_window,wrap,TAU
from e01 import estimate
from geometry_adapter import full_layout,local_support
from run_e04 import window_run

def integer_sync(n,edges,offsets):
    """min sum |k_j-k_i+round(mu_ij)|, gauge min node=0/component.

    Incidence matrix with +/- slack is totally unimodular, integer RHS;
    HiGHS dual simplex returns integral vertex. Verify rather than assume.
    """
    if not edges: return np.zeros(n,int),np.arange(n),dict(objective=0.,integrality_error=0.)
    edges=np.asarray(edges,int); m=len(edges)
    B=coo_matrix((np.tile([-1.,1.],m),(np.repeat(np.arange(m),2),edges.ravel())),shape=(m,n)).tocsr()
    G=coo_matrix((np.ones(2*m),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n))
    count,comp=connected_components(G,directed=False)
    bounds=[(None,None)]*n+[(0,None)]*(2*m)
    for c in range(count): bounds[int(np.flatnonzero(comp==c)[0])]=(0,0)
    d=np.rint(offsets)
    result=linprog(np.r_[np.zeros(n),np.ones(2*m)],A_eq=hstack([B,eye(m),-eye(m)]).tocsr(),b_eq=-d,bounds=bounds,method='highs-ds')
    if not result.success: raise RuntimeError(result.message)
    discrepancy=float(np.max(abs(result.x[:n]-np.rint(result.x[:n]))))
    if discrepancy>1e-7: raise RuntimeError('Nonintegral LP optimum; no silent rounding')
    return np.rint(result.x[:n]).astype(int),comp,dict(objective=float(result.fun),integrality_error=discrepancy)

def hex_outputs(phi,mask,orientation):
    start=time.perf_counter(); labels,centers,inc,edges=full_layout(phi.shape,orientation=orientation)
    es=local_support(labels,centers,edges,mask,inc); setup=time.perf_counter()-start
    for wm in ['W0','W1_MLSQU_HEX_Q2']:
        tick=time.perf_counter(); raw,diagnostics=window_run(phi,mask,labels,centers,wm); localtime=time.perf_counter()-tick
        accepted={i:d['valid'] for i,d in diagnostics}
        measured={'M0':[],'M1':[]}; all_edges=[]
        for eid,e in enumerate(es):
            ok=accepted[e['i']] and accepted[e['j']] and len(e['pairs'])
            vals=estimate(raw,phi,e,1.) if ok else [(pm,np.nan,np.nan,np.nan,False,0) for pm in ['M0','M1']]
            for pm,mu,J,Q,valid,retained in vals:
                if valid and np.isfinite(mu): measured[pm].append((e['i'],e['j'],mu))
                all_edges.append(dict(edge_id=eid,i=e['i'],j=e['j'],pair_method=pm,mu=mu,valid=bool(valid),J_unit_reference=J,Q_unit_reference=Q))
        pairtime=time.perf_counter()-tick-localtime
        for pm in ['M0','M1']:
            graphstart=time.perf_counter(); obs=measured[pm]
            k,comp,diag=integer_sync(len(centers),[(i,j) for i,j,d in obs],[d for i,j,d in obs])
            phase=raw+TAU*k[labels]; pixelcomp=np.where(np.isfinite(phase),comp[labels]+1,0)
            yield f'{orientation}_{wm}_{pm}',phase,pixelcomp,dict(runtime=setup+localtime+pairtime+time.perf_counter()-graphstart,accepted_windows=sum(accepted.values()),total_windows=len(centers),accepted_edges=len(obs),total_edges=len(es),graph_components=len(np.unique(pixelcomp[pixelcomp>0])),**diag),all_edges

def spatial_bfs(phi,mask):
    start=time.perf_counter(); comp,n=label(mask); result=np.full(phi.shape,np.nan)
    yy,xx=np.indices(phi.shape); coords=np.stack([xx,yy],-1)
    for c in range(1,n+1):
        region=comp==c; psi,d=unwrap_window(phi,region,coords,dict(method='W0'))
        result[region]=psi[region]
    return result,comp,dict(runtime=time.perf_counter()-start,graph_components=n)

def hierarchical(phases,frequencies):
    """Standard hierarchical TPU. Positive principal lowest phase, no oracle."""
    result=np.mod(phases[frequencies[0]],TAU)
    for a,b in zip(frequencies[:-1],frequencies[1:]):
        high=phases[b]; result=high+TAU*np.rint((result*b/a-high)/TAU)
    return result
