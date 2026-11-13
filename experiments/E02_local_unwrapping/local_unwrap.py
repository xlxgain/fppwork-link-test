"""Observation-only window algorithms. No truth, simulator or file access."""
from collections import deque
from time import perf_counter
import numpy as np
from scipy.ndimage import label

TAU=2*np.pi
def wrap(x): return np.angle(np.exp(1j*x))

def penalty(r,edges):
    values=[np.mean(abs(r[b]-r[a])) for a,b in edges if len(a)]
    return sum(values) if len(values)==2 else np.inf

def unwrap_window(phi,window_mask,coords,config):
    start=perf_counter(); mask=np.asarray(window_mask,bool)
    ids=np.flatnonzero(mask); out=np.full(phi.shape,np.nan)
    _,components=label(mask)
    d=dict(valid=False,reason='',n_components=components,model_rank=0,condition=np.nan,rho_grid_index=-1,min_cost=np.nan,tie_count=0,runtime=0.)
    def finish(reason):
        d['reason']=reason; d['runtime']=perf_counter()-start
        return out,d
    if not len(ids): return finish('empty')
    if components!=1: return finish('disconnected/needs_multiple_offsets')
    values=phi.ravel()[ids]
    if not np.isfinite(values).all(): return finish('nonfinite_observation')
    lookup=np.full(phi.size,-1,int); lookup[ids]=np.arange(len(ids)); lookup=lookup.reshape(phi.shape)
    edges=[]
    for aa,bb in [(lookup[:,:-1],lookup[:,1:]),(lookup[:-1],lookup[1:])]:
        ok=(aa>=0)&(bb>=0); edges.append((aa[ok],bb[ok]))
    xy=np.asarray(coords).reshape(-1,2)[ids]
    if config['method']=='W0':
        center=np.asarray(config.get('center',xy.mean(axis=0)))
        root=int(np.argmin(np.sum((xy-center)**2,axis=1)))
        neighbors=[[] for _ in ids]
        # fixed traversal order: increasing row-major neighbor index
        for a,b in edges:
            for u,v in zip(a,b): neighbors[u].append(v); neighbors[v].append(u)
        k=np.zeros(len(ids),int); seen=np.zeros(len(ids),bool); seen[root]=True; queue=deque([root])
        while queue:
            a=queue.popleft()
            for b in sorted(neighbors[a],key=lambda v:ids[v]):
                if not seen[b]:
                    k[b]=k[a]+int(np.rint((values[a]+wrap(values[b]-values[a])-values[b])/TAU))
                    seen[b]=True; queue.append(b)
        result=values+TAU*k
        d['root_index']=int(ids[root]); d['traversal']='BFS; row-major neighbor order'
    else:
        if any(not len(a) for a,b in edges): return finish('missing_direction')
        scale=xy.std(axis=0)
        if np.any(scale==0): return finish('rank_deficient')
        x,y=((xy-xy.mean(axis=0))/scale).T
        V=np.column_stack([x**a*y**b for a in range(3) for b in range(3) if a+b])
        D=np.vstack([V[b]-V[a] for a,b in edges]); g=np.concatenate([wrap(values[b]-values[a]) for a,b in edges])
        c,_,rank,s=np.linalg.lstsq(D,g,rcond=1e-12)
        d.update(model_rank=int(rank),condition=float(s[0]/s[-1]) if s[-1]>0 else np.inf)
        if rank!=8 or d['condition']>1e10: return finish('rank_deficient_or_ill_conditioned')
        f=V@c; rho=TAU*np.arange(40)/40
        costs=np.array([penalty(wrap(values-f-r)+r,edges) for r in rho])
        ix=int(np.argmin(costs)); ties=np.flatnonzero(np.isclose(costs,costs[ix],rtol=0,atol=1e-12))
        ix=int(ties[0]); candidate=f+rho[ix]+wrap(values-f-rho[ix])
        result=values+TAU*np.rint((candidate-values)/TAU)
        d.update(rho_grid_index=ix,min_cost=float(costs[ix]),tie_count=len(ties))
    out.ravel()[ids]=result; d['valid']=True
    return finish('')
