"""Truth-assisted scoring only, never used to select algorithm outputs."""
import numpy as np
TAU=2*np.pi
def wrap(x): return np.angle(np.exp(1j*x))

def evaluate(psi,phi,truth,mask,diag,zones):
    n=int(mask.sum()); ok=mask&np.isfinite(psi); count=int(ok.sum()); missing=n-count
    result=dict(N_all=n,N_out=count,N_missing=missing,N_bad_out=0,c=np.nan,coverage=count/n if n else np.nan,covered_relative_error_rate=np.nan,total_failure_rate=missing/n if n else np.nan,strict_success=False,continuous_rmse=np.nan,wrap_consistency_error=np.nan,status='not_applicable' if not n else 'algorithm_rejected')
    residual=np.full(phi.shape,np.nan)
    if count:
        T=truth+wrap(phi-wrap(truth)); e=np.rint((psi[ok]-T[ok])/TAU).astype(int)
        vals,counts=np.unique(e,return_counts=True); candidates=vals[counts==counts.max()]
        c=int(sorted(candidates,key=lambda v:(abs(v),v))[0]); bad=int(sum(e!=c)); residual[ok]=e-c
        success=bool(diag['valid'] and diag['n_components']==1 and count==n and bad==0)
        result.update(c=c,N_bad_out=bad,covered_relative_error_rate=bad/count,total_failure_rate=(bad+missing)/n,strict_success=success,continuous_rmse=float(np.sqrt(np.mean((psi[ok]-TAU*c-truth[ok])**2))),wrap_consistency_error=float(np.max(abs(wrap(psi[ok]-phi[ok])))),status='correct' if success else 'returned_integer_error_or_partial' if diag['valid'] else 'algorithm_rejected')
    for zone,name in enumerate(['core','transition','outside']):
        selected=mask&(zones==zone); covered=selected&ok
        result.update({f'{name}_N_all':int(selected.sum()),f'{name}_N_out':int(covered.sum()),f'{name}_N_bad_out':int(np.sum(residual[covered]!=0)),f'{name}_N_missing':int(np.sum(selected&~ok))})
    return result,residual

def reference(si,sj):
    both=si['strict_success'] and sj['strict_success']
    return dict(both_windows_correct=both,reference_status='unique' if both else 'input_locally_inconsistent',d_reference=sj['c']-si['c'] if both else np.nan)

def observation(phi,truth,Bhat,Btrue,mask,zones):
    error=wrap(phi-truth); rows=[]
    for zone,name in [(-1,'all'),(0,'core'),(1,'transition'),(2,'outside')]:
        selected=mask if zone==-1 else mask&(zones==zone); e=error[selected]
        rows.append(dict(zone=name,n_pixels=len(e),squared_error_sum=float(e@e),circular_rmse=float(np.sqrt(np.mean(e*e))),absolute_p50=float(np.quantile(abs(e),.5)),absolute_p95=float(np.quantile(abs(e),.95)),over_pi_half_count=int(sum(abs(e)>np.pi/2)),over_pi_half_fraction=float(np.mean(abs(e)>np.pi/2)),B_true_mean=float(Btrue[selected].mean()),B_hat_mean=float(Bhat[selected].mean())))
    return rows

def propagation(edge,Btrue,sigma):
    if not edge['valid']: return np.nan,'invalid_support'
    if sigma==0: return 0.,'no_noise'
    b=Btrue.ravel()[edge['ix']]; q=edge['residual']/(TAU*edge['info'])
    return float(np.sum(q*q*sigma*sigma/(2*b*b))),'higher_snr' if b.min()/sigma>=5 else 'weak_signal'
