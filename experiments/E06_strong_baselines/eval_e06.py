"""One input-mask component gauge, then ALL ROI use the SAME residual map."""
import numpy as np
from scipy.ndimage import label
TAU=2*np.pi
def wrap(x): return np.angle(np.exp(1j*x))

def evaluate(psi,phi,truth,mask,zones):
    finite=mask&np.isfinite(psi); integer=np.full(mask.shape,np.nan)
    T=truth+wrap(phi-wrap(truth)); integer[finite]=np.rint((psi[finite]-T[finite])/TAU)
    residual=np.full(mask.shape,np.nan); aligned=np.full(mask.shape,np.nan); comp,n=label(mask); success=0; gauges=[]
    for c in range(1,n+1):
        region=comp==c; ok=region&finite
        if ok.any():
            vals,counts=np.unique(integer[ok],return_counts=True); winners=vals[counts==counts.max()]; gauge=int(sorted(winners,key=lambda x:(abs(x),x))[0]); gauges.append(gauge)
            residual[ok]=integer[ok]-gauge; aligned[ok]=psi[ok]-TAU*gauge
        success+=int(ok.sum()==region.sum() and np.all(residual[ok]==0))
    rois=[]
    for name,region in [('all',mask),('core',mask&(zones==0)),('transition',mask&(zones==1)),('outside',mask&(zones==2)),('patch',mask&(zones!=2))]:
        ok=region&finite; N=int(region.sum()); Nout=int(ok.sum()); bad=int(np.sum(residual[ok]!=0)); missing=N-Nout
        rois.append(dict(roi=name,N_all=N,N_out=Nout,N_missing=missing,N_bad_out=bad,relative_integer_error_total=(bad+missing)/N if N else np.nan,covered_error=bad/Nout if Nout else np.nan,coverage_common=Nout/N if N else np.nan,continuous_squared_error_sum=float(np.sum((aligned[ok]-truth[ok])**2)),relative_phase_rmse=float(np.sqrt(np.mean((aligned[ok]-truth[ok])**2))) if Nout else np.nan,circular_rmse=float(np.sqrt(np.mean(wrap(phi[region]-truth[region])**2))) if N else np.nan))
    result=rois[0].copy(); result.pop('roi')
    result.update(coverage_all_image=float(finite.mean()),component_success=success/n if n else np.nan,whole_image_success=bool(finite.sum()==mask.sum() and np.all(residual[finite]==0)),raw_absolute_bad=int(np.sum(integer[finite]!=0)),raw_absolute_zero_modes=str(gauges),raw_absolute_correct=float(np.sum(integer[finite]==0)/mask.sum()),wrap_closure=float(np.max(abs(wrap(psi[finite]-phi[finite])))) if finite.any() else np.nan)
    anchor=(128,128); result['anchor_available']=bool(finite[anchor]) if min(mask.shape)>128 else False
    result['single_external_anchor_correct']=float(np.sum(integer[finite]==integer[anchor])/mask.sum()) if result['anchor_available'] else np.nan
    return result,rois,integer,residual,aligned
