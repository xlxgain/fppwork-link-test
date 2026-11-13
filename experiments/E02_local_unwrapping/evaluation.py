"""Truth permitted only for scoring; never feed aligned outputs to estimators."""
import numpy as np
from local_unwrap import TAU,wrap

def evaluate(psi,phi,truth,mask,diagnostics):
    n=int(mask.sum()); ok=mask&np.isfinite(psi)
    result=dict(relative_integer_error_rate=np.nan,strict_success=False,c=0,coverage=float(ok.sum()/n) if n else 0.,wrap_consistency_error=np.nan,continuous_rmse=np.nan)
    residual=np.full(phi.shape,np.nan)
    if not ok.any(): return result,residual
    reference=truth+wrap(phi-wrap(truth))
    e=np.rint((psi[ok]-reference[ok])/TAU).astype(int)
    values,counts=np.unique(e,return_counts=True)
    winners=values[counts==counts.max()]; c=int(sorted(winners,key=lambda v:(abs(v),v))[0])
    residual[ok]=e-c
    rate=float(np.mean(e!=c))
    result.update(relative_integer_error_rate=rate,c=c,strict_success=bool(diagnostics['valid'] and diagnostics['n_components']==1 and ok.sum()==n and rate==0),wrap_consistency_error=float(np.max(abs(wrap(psi[ok]-phi[ok])))),continuous_rmse=float(np.sqrt(np.mean((psi[ok]-TAU*c-truth[ok])**2))))
    return result,residual
