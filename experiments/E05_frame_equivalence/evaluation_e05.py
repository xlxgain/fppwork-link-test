"""Same observed spatial components for every method, never method-dependent alignment."""
import numpy as np
from scipy.ndimage import label
from algorithms import TAU,wrap

def mode(values):
    vals,counts=np.unique(values,return_counts=True); winners=vals[counts==counts.max()]
    return int(sorted(winners,key=lambda v:(abs(v),v))[0])

def evaluate(phase,phi,truth,mask,anchor_index):
    finite=mask&np.isfinite(phase); n=int(mask.sum()); integer=np.full(mask.shape,np.nan)
    T=truth+wrap(phi-wrap(truth)); integer[finite]=np.rint((phase[finite]-T[finite])/TAU)
    comp,count=label(mask); residual=np.full(mask.shape,np.nan); successes=0
    for c in range(1,count+1):
        region=comp==c; ok=region&finite
        if ok.any(): residual[ok]=integer[ok]-mode(integer[ok].astype(int))
        successes+=int(ok.sum()==region.sum() and np.all(residual[ok]==0))
    covered=int(finite.sum()); bad=int(np.sum(abs(residual[finite])>0)); absbad=int(np.sum(abs(integer[finite])>0))
    # Explicit separate calibration experiment, one GLOBAL scalar only.
    anchored=integer.copy(); a=np.unravel_index(anchor_index,mask.shape)
    anchor_available=bool(finite[a])
    if anchor_available: anchored[finite]-=integer[a]
    else: anchored[:]=np.nan
    result=dict(N_common=n,N_out=covered,coverage_common=covered/n,coverage_all=covered/mask.size,relative_bad=bad,relative_correct_all_common=(covered-bad)/n,relative_component_success=successes/count if count else np.nan,absolute_bad=absbad,absolute_correct_all_common=(covered-absbad)/n,absolute_rmse=float(np.sqrt(np.mean((phase[finite]-truth[finite])**2))) if covered else np.nan,relative_rmse=float(np.sqrt(np.mean((phase[finite]-TAU*(integer[finite]-residual[finite])-truth[finite])**2))) if covered else np.nan,fractional_phase_rmse=float(np.sqrt(np.mean(wrap(phi[mask]-truth[mask])**2))),single_external_anchor_available=anchor_available,anchored_correct_all_common=float(np.sum(anchored[finite]==0)/n) if anchor_available else np.nan,wrap_closure=float(np.max(abs(wrap(phase[finite]-phi[finite])))) if covered else np.nan,common_components=count)
    return result,integer,residual
