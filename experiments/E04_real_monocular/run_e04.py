import sys,json,time,hashlib,subprocess,platform
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parent; E02=ROOT.parent/'E02_local_unwrapping'; E01=ROOT.parent/'E01_hex_pairwise'
sys.path.insert(0,str(E01)); sys.path.insert(0,str(E02))
from local_unwrap import unwrap_window,wrap,TAU
from e01 import estimate,demodulate,plt
from geometry_adapter import full_layout,local_support

def phase_maps(images):
    images=np.asarray(images,dtype=float)
    if images.shape[0]!=4: raise ValueError('Exactly four frames required')
    return demodulate(images),images.mean(axis=0),.5*np.hypot(images[0]-images[2],images[3]-images[1])

def temporal_candidate(phases,counts):
    result=phases[0].copy(); diagnostics=[]
    for k in range(1,len(counts)):
        predicted=result*counts[k]/counts[k-1]
        residual=wrap(phases[k]-predicted)
        diagnostics.append(residual)
        result=phases[k]+TAU*np.rint((predicted-phases[k])/TAU)
    return result,diagnostics

def window_run(phi,mask,labels,centers,method):
    raw=np.full(phi.shape,np.nan); records=[]; h,w=phi.shape
    for i,center in enumerate(centers):
        y,x=np.nonzero(labels==i)
        x0,x1=x.min(),x.max()+1; y0,y1=y.min(),y.max()+1
        yy,xx=np.indices((y1-y0,x1-x0)); coords=np.stack([xx+x0,yy+y0],-1)
        region=(labels[y0:y1,x0:x1]==i)&mask[y0:y1,x0:x1]
        psi,diag=unwrap_window(phi[y0:y1,x0:x1],region,coords,dict(method=method,center=center))
        localphi=phi[y0:y1,x0:x1]; jumps=[]
        for a,b,pa,pb,valid in [(psi[:,:-1],psi[:,1:],localphi[:,:-1],localphi[:,1:],region[:,:-1]&region[:,1:]),(psi[:-1],psi[1:],localphi[:-1],localphi[1:],region[:-1]&region[1:])]:
            good=valid&np.isfinite(a)&np.isfinite(b)
            jumps.extend(np.rint((b[good]-a[good]-wrap(pb[good]-pa[good]))/TAU))
        diag['neighbor_integer_jump_fraction']=float(np.mean(np.asarray(jumps)!=0)) if len(jumps) else np.nan
        diag['covered_neighbor_edges']=len(jumps)
        patch=raw[y0:y1,x0:x1]; patch[region]=psi[region]
        records.append((i,diag))
    return raw,records

def run():
    start=time.perf_counter(); cfg=json.loads((ROOT/'config.json').read_text()); out=ROOT/'outputs'; out.mkdir(exist_ok=True)
    cache=ROOT/'data_cache'; version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cache,text=True).strip(); assert version==cfg['source_commit']
    files=[]; maps={}; stacks={}; checks=[]
    for kind,groups in [('reference',cfg['reference_groups']),('object',cfg['object_groups'])]:
        for k,ids in enumerate(groups):
            images=[]
            for step,idx in enumerate(ids):
                p=cache/'data/sphere'/f'DSC_{idx:04d}.JPG'; im=Image.open(p)
                assert im.format=='JPEG' and im.size==(1152,768)
                files.append(dict(kind=kind,frequency_index=k,step=step,phase_shift_rad=step*np.pi/2,file=str(p.relative_to(cache)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=list(im.size),mode=im.mode))
                images.append(np.asarray(im.convert('L'),dtype=float))
            images=np.array(images); stacks[kind,k]=images
            phi,A,B=phase_maps(images); maps[kind,k]=(phi,A,B)
            fit=A[None]+B[None]*np.cos(phi[None]+np.arange(4)[:,None,None]*np.pi/2)
            checks.append(dict(kind=kind,frequency_index=k,fit_rmse=float(np.sqrt(np.mean((images-fit)**2))),opposite_sum_rmse=float(np.sqrt(np.mean(((images[0]+images[2]-images[1]-images[3])/2)**2))),fft_central_strip_fullwidth=fft_count(images[0])))
    phi,A,B=maps['object',3]; ref,Ar,Br=maps['reference',3]
    threshold=cfg['modulation_fraction_p99']*np.percentile(B,99)
    exposure=(A>cfg['dc_min'])&(A<cfg['dc_max']); mask=exposure&(B>threshold)&np.isfinite(phi)
    low=exposure&(B<=threshold)
    yy,xx=np.indices(phi.shape); radius=np.hypot(xx-cfg['sphere_display_center_xy'][0],yy-cfg['sphere_display_center_xy'][1])
    boundary=abs(radius-cfg['sphere_display_radius_px'])<=cfg['edge_band_halfwidth_px']; sphere=radius<cfg['sphere_display_radius_px']
    # Independent diagnostic ONLY: source works on object-reference differences.
    diffs=[wrap(maps['object',k][0]-maps['reference',k][0]) for k in range(4)]
    delta,residuals=temporal_candidate(diffs,cfg['fringe_counts_sphere'])
    ref_carrier=np.unwrap(np.unwrap(ref,axis=1),axis=0)
    candidate=ref_carrier+delta
    closure=abs(wrap(candidate-phi))
    # Algebraic congruence is necessary but NOT sufficient. No calibrated absolute
    # order or certified frequency origins supplied: no true accuracy claim.
    trusted=np.zeros(phi.shape,bool)
    reference_rows=[]
    for k,res in enumerate(residuals,1):
        valid=mask.copy()
        for level in range(k+1): valid&=(maps['object',level][2]>threshold)&(maps['reference',level][2]>threshold)
        reference_rows.append(dict(stage=k,ratio=cfg['fringe_counts_sphere'][k]/cfg['fringe_counts_sphere'][k-1],n=int(valid.sum()),prediction_circular_rmse=float(np.sqrt(np.mean(res[valid]**2))),fraction_abs_residual_under_02=float(np.mean(abs(res[valid])<.2)),trusted_pixels=0,status='unverified_frequency_origin_and_lowest_order'))
    windows=[]; pairs=[]; summaries=[]
    for orientation in ['flat','pointy']:
        labels,c,inc,edges=full_layout(phi.shape,orientation=orientation)
        es=local_support(labels,c,edges,mask,inc)
        for wm in ['W0','W1_MLSQU_HEX_Q2']:
            raw,diags=window_run(phi,mask,labels,c,wm); valid_windows={i:d['valid'] for i,d in diags}
            for i,d in diags:
                region=(labels==i)&mask; ok=region&np.isfinite(raw); n=int(region.sum()); nout=int(ok.sum())
                e=np.rint((raw[ok]-candidate[ok])/TAU).astype(int)
                vals,counts=np.unique(e,return_counts=True); mode=int(vals[np.argmax(counts)]) if len(e) else 0
                candidate_inconsistency=float(np.mean(e!=mode)) if len(e) else np.nan
                windows.append(dict(orientation=orientation,window_method=wm,window_id=i,connection='truncated' if inc[i] else 'internal',N_geometric=int(np.sum(labels==i)),N_all=n,N_out=nout,N_missing=n-nout,coverage=nout/n if n else np.nan,wrap_closure=float(np.max(abs(wrap(raw[ok]-phi[ok])))) if nout else np.nan,candidate_integer_inconsistency=candidate_inconsistency,true_integer_error_rate=np.nan,reference_status='unverified',boundary_pixels=int(np.sum(region&boundary)),low_modulation_geometric_pixels=int(np.sum((labels==i)&low)),**d))
            for edge_id,e in enumerate(es):
                tick=time.perf_counter(); i,j=e['i'],e['j']; usable=valid_windows[i] and valid_windows[j] and len(e['pairs'])>0
                values=estimate(raw,phi,e,1.) if usable else [(m,np.nan,np.nan,np.nan,False,0) for m in ['M0','M1']]
                for pm,mu,J,Q,valid,retained in values:
                    pairs.append(dict(orientation=orientation,window_method=wm,pair_method=pm,edge_id=edge_id,i=i,j=j,direction=e['direction'],connection=e['connection'],mu=mu,J_unit_reference=J,Q_unit_reference=Q,uncertainty_status='unit_geometry_only_not_calibrated',valid=valid,reason='' if valid else 'window_failure' if not(valid_windows[i] and valid_windows[j]) else 'empty_support' if not len(e['pairs']) else 'rank_or_condition',d_reference=np.nan,true_integer_error_rate=np.nan,reference_status='unverified',support_pixels=len(e['ix']),crossing_pairs=len(e['pairs']),rank=e['rank'],condition=e['condition'],retained_fraction=retained,boundary_related=bool(np.any(boundary.ravel()[e['ix']])) if len(e['ix']) else False,runtime=time.perf_counter()-tick))
            finite=mask&np.isfinite(raw)
            fig,axs=plt.subplots(1,3,figsize=(15,4))
            axs[0].imshow(np.where(mask,labels%20,np.nan),cmap='tab20',vmin=0,vmax=19); axs[0].set_title(orientation+' valid hex labels')
            axs[1].imshow(raw,cmap='viridis'); axs[1].set_title(wm+' raw local output')
            shift=np.where(finite,np.rint((raw-candidate)/TAU),np.nan); axs[2].imshow(shift,cmap='coolwarm'); axs[2].set_title('Candidate order difference (NOT truth)')
            for ax in axs: ax.set_xlim(400,730); ax.set_ylim(560,280)
            fig.tight_layout(); fig.savefig(out/f'local_{orientation}_{wm}.png',dpi=130); plt.close(fig)
            print(orientation,wm,'complete',flush=True)
    w=pd.DataFrame(windows); p=pd.DataFrame(pairs)
    w.to_csv(out/'windows.csv',index=False); p.to_csv(out/'pairs.csv',index=False)
    sw=w.groupby(['orientation','window_method','connection']).agg(windows=('valid','size'),accepted=('valid','sum'),N_all=('N_all','sum'),N_out=('N_out','sum'),N_missing=('N_missing','sum'),max_wrap_closure=('wrap_closure','max'),candidate_inconsistency_mean=('candidate_integer_inconsistency','mean')).reset_index(); sw['coverage']=sw.N_out/sw.N_all; sw.to_csv(out/'summary_windows.csv',index=False)
    sp=p.groupby(['orientation','window_method','pair_method','connection','boundary_related']).agg(edges=('valid','size'),outputs=('valid','sum'),mu_mean=('mu','mean'),mu_std=('mu','std')).reset_index(); sp['valid_fraction']=sp.outputs/sp.edges; sp.to_csv(out/'summary_pairs.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'capture_checks.csv',index=False); pd.DataFrame(reference_rows).to_csv(out/'reference_checks.csv',index=False)
    observations=[]
    for name,region in [('all',np.ones(mask.shape,bool)),('valid',mask),('low_modulation',low),('sphere_boundary_display',boundary),('sphere_interior_display',sphere&~boundary)]:
        observations.append(dict(zone=name,pixels=int(region.sum()),valid_pixels=int((region&mask).sum()),A_mean=float(A[region].mean()),B_mean=float(B[region].mean()),B_p10=float(np.percentile(B[region],10)),closure_mean=float(closure[region].mean())))
    pd.DataFrame(observations).to_csv(out/'observation_zones.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,array,title in zip(axs,[stacks['object',3][0],B,low.astype(int)+2*boundary.astype(int)],['Native sphere / edge view','Modulation near sphere edge','Low modulation + display boundary']):
        ax.imshow(array,cmap='gray' if title.startswith('Native') else 'viridis'); ax.set_xlim(420,720); ax.set_ylim(550,290); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out/'edge_low_modulation.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,7)); ax.imshow(np.where(mask,labels%20,np.nan),cmap='tab20',vmin=0,vmax=19); ax.set_title('Full native-pixel hex partition (pointy)'); fig.tight_layout(); fig.savefig(out/'full_hex_layout.png',dpi=120); plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for ax,array,title,cmap in zip(axs.ravel(),[stacks['object',3][0],phi,A,B,mask,low|boundary],['Real JPEG fringe','Wrapped phase','DC brightness','Modulation','Valid mask','Low modulation / display edge'],['gray','twilight','gray','viridis','gray','gray']):
        im=ax.imshow(array,cmap=cmap); ax.set_title(title); fig.colorbar(im,ax=ax,shrink=.6)
    fig.tight_layout(); fig.savefig(out/'real_observations.png',dpi=140); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for (ori,wm,pm),g in p.groupby(['orientation','window_method','pair_method']):
        axs[0].plot(g.edge_id,g.mu,'.',markersize=2,label=ori+' '+wm+' '+pm)
    axs[0].set(xlabel='Edge ID',ylabel='Offset estimate (cycles)'); axs[0].legend(fontsize=5)
    im=axs[1].imshow(np.where(mask,delta,np.nan),cmap='viridis'); axs[1].set_title('Multi-frequency difference candidate'); fig.colorbar(im,ax=axs[1],shrink=.6)
    fig.tight_layout(); fig.savefig(out/'pair_and_reference.png',dpi=140); plt.close(fig)
    np.savez_compressed(out/'lightweight_profiles.npz',row=384,phi=phi[384],A=A[384],B=B[384],candidate=candidate[384],valid=mask[384])
    manifest=dict(source=cfg['source'],source_commit=version,baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=files,source_code_hashes={name:hashlib.sha256((cache/name).read_bytes()).hexdigest() for name in ['README.md','src/sphere_multifreq.py','src/phase_shifting.py','src/measure_freqs.py','LICENSE']},environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),no_additional_resampling=True,shape=list(phi.shape),algorithm_sources='E02/local_unwrap.py; E01/e01.py and e01_c.py support',phase_definition='absolute object phase at highest frequency; multifrequency difference converted using independent unverified reference carrier',trusted_integer_reference_pixels=0)
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    stats=dict(runtime_seconds=time.perf_counter()-start,valid_pixels=int(mask.sum()),total_pixels=mask.size,window_records=len(w),pair_records=len(p),trusted_reference_pixels=0,max_candidate_modulo_error=float(closure.max()))
    (out/'run_stats.json').write_text(json.dumps(stats,indent=2)); print(stats)

def fft_count(image):
    h,w=image.shape; strip=image[h//2-50:h//2+50].mean(axis=0); strip=strip-strip.mean(); strip=strip[w//4:3*w//4]
    spectrum=abs(np.fft.rfft(strip)); spectrum[0]=0
    return float(np.argmax(spectrum)*w/len(strip))

if __name__=='__main__': run()
