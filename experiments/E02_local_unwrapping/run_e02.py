import sys,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'E01_hex_pairwise'
sys.path.insert(0,str(OLD))
from e01 import layout,estimate,demodulate,plt
from e01_c import support
from local_unwrap import unwrap_window,wrap,TAU
from evaluation import evaluate

def input_images(truth,config,scene,sigma,repeat):
    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],scene,int(sigma),repeat]))
    images=config['A']+config['B']*np.cos(truth[None]+np.arange(4)[:,None,None]*np.pi/2)
    images+=rng.normal(0,sigma,images.shape)
    return images,hashlib.sha256(images.tobytes()).hexdigest()

def run():
    start=time.perf_counter(); cfg=json.loads((ROOT/'config.json').read_text())
    out=ROOT/'outputs_E02'; review=ROOT/'outputs_C_review'
    out.mkdir(exist_ok=True); review.mkdir(exist_ok=True)
    manifest=dict(config=cfg,baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),geometry=json.loads((OLD/'parameters_C.json').read_text()),scene_definition='fixed flat reference grid; center/edge/junction labels apply ONLY to reference flat grid',inputs=[],scenes={})
    windows=[]; pairs=[]; oracle=[]; yy,xx=np.indices((256,256)); coords=np.stack([xx,yy],axis=-1)
    for si,scene in enumerate(['plane','center','edge','junction']):
        source=OLD/'outputs_C'/f'flat_{scene}.npz'; frozen=np.load(source)
        truth=frozen['phase_radians']; mask=frozen['valid']
        # Preserve exact historical arrays without resampling or repositioning.
        np.savez_compressed(review/f'{scene}.npz',**dict(frozen))
        manifest['scenes'][scene]=dict(source=str(source.relative_to(ROOT.parent)),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        grids={}
        for orientation in ['flat','pointy']:
            labels,c,inc,edges=layout(orientation=orientation)
            grids[orientation]=(labels,c,inc,support(labels,c,edges,mask,inc))
        for sigma in cfg['sigma']:
            for repeat in range(1 if sigma==0 else cfg['repeats']):
                images,digest=input_images(truth,cfg,si,sigma,repeat); phi=demodulate(images)
                manifest['inputs'].append(dict(scene_id=scene,sigma=sigma,repeat=repeat,input_hash=digest))
                T=truth+wrap(phi-wrap(truth))
                for orientation,(labels,c,inc,es) in grids.items():
                    common=dict(scene_id=scene,input_hash=digest,orientation=orientation,sigma=sigma,repeat=repeat)
                    for wm in ['W0','W1_MLSQU_HEX_Q2','oracle_control']:
                        raw=np.full(phi.shape,np.nan); scores={}; diagnostics={}; residualmap=np.full(phi.shape,np.nan)
                        for i in range(len(c)):
                            region=(labels==i)&mask
                            if wm=='oracle_control':
                                # Separate stream for reference integers; unrelated to observations.
                                m=np.random.default_rng(cfg['reference_seed']+i).integers(-8,9)
                                psi=np.where(region,T-TAU*m,np.nan)
                                diag=dict(valid=bool(region.any()),reason='',n_components=1,model_rank=0,condition=np.nan,rho_grid_index=-1,min_cost=np.nan,tie_count=0,runtime=0.)
                            else:
                                psi,diag=unwrap_window(phi,region,coords,dict(method=wm,center=c[i]))
                            score,residual=evaluate(psi,phi,truth,region,diag)
                            raw[region]=psi[region]; residualmap[region]=residual[region]
                            scores[i]=score; diagnostics[i]=diag
                            if wm!='oracle_control':
                                windows.append(dict(**common,window_id=i,window_method=wm,internal_or_truncated='truncated' if inc[i] else 'internal',n_pixels=int(region.sum()),**diag,**score))
                        if sigma==5 and repeat==0 and orientation=='flat' and scene=='edge' and wm!='oracle_control':
                            np.savez_compressed(out/f'example_{wm}.npz',phi=phi,psi=raw,residual_integer_error=residualmap)
                            fig,ax=plt.subplots(1,2,figsize=(9,4)); ax[0].imshow(raw); ax[0].set_title(wm+' raw output'); ax[1].imshow(residualmap,vmin=-1,vmax=1,cmap='coolwarm'); ax[1].set_title('Evaluated internal integer error'); fig.tight_layout(); fig.savefig(out/f'example_{wm}.png'); plt.close(fig)
                        for edge_id,e in enumerate(es):
                            tick=time.perf_counter(); i,j=e['i'],e['j']; both=scores[i]['strict_success'] and scores[j]['strict_success']
                            d=scores[j]['c']-scores[i]['c']
                            inputs_valid=diagnostics[i]['valid'] and diagnostics[j]['valid']
                            vals=estimate(raw,phi,e,sigma*sigma/20000 if sigma else 1) if inputs_valid and len(e['pairs']) else [(m,np.nan,np.nan,np.nan,False,0.) for m in ['M0','M1']]
                            for pm,mu,J,Q,valid,retained in vals:
                                reason='' if valid else ('window_failure' if not inputs_valid else 'empty_support' if not len(e['pairs']) else 'rank_or_condition_failure')
                                row=dict(**common,i=i,j=j,edge_id=edge_id,direction=e['direction'],connection=e['connection'],method=pm,window_method=wm,pair_method=pm,d_reference=d if both else np.nan,modal_offset_diagnostic=d,mu=mu,J=J,Q=Q,error=mu-d if both and valid else np.nan,integer_error=float(round(mu)!=d) if both and valid else np.nan,valid=valid,invalid_reason=reason,support_pixel_count=len(e['ix']),crossing_pair_count=len(e['pairs']),rank=e['rank'],condition=e['condition'],retained_fraction=retained,runtime=time.perf_counter()-tick,both_windows_correct=both,reference_status='unique' if both else 'input_locally_inconsistent',phase_variance_assumption='unit_reference' if sigma==0 else 'first_order_independent_intensity')
                                (oracle if wm=='oracle_control' else pairs).append(row)
    w=pd.DataFrame(windows); p=pd.DataFrame(pairs); o=pd.DataFrame(oracle)
    w.to_csv(out/'windows.csv',index=False); p.to_csv(out/'pairs.csv',index=False); o.to_csv(review/'pairs.csv',index=False)
    wk=['orientation','scene_id','sigma','window_method','internal_or_truncated']
    w['bad_pixels']=w.relative_integer_error_rate.fillna(1)*w.n_pixels
    w['covered_pixels']=w.coverage*w.n_pixels
    sw=w.groupby(wk).agg(windows=('valid','size'),failures=('valid',lambda v:int((~v).sum())),strict_success_rate=('strict_success','mean'),pixels=('n_pixels','sum'),bad_pixels=('bad_pixels','sum'),covered_pixels=('covered_pixels','sum'),continuous_rmse=('continuous_rmse','mean')).reset_index()
    sw['pixel_weighted_internal_error']=sw.bad_pixels/sw.pixels; sw['coverage']=sw.covered_pixels/sw.pixels; sw.to_csv(out/'summary_windows.csv',index=False)
    def summarize(frame,path):
        keys=['orientation','scene_id','sigma','window_method','pair_method','connection','direction']
        s=frame.groupby(keys).agg(geometric_edges=('valid','size'),valid_fraction=('valid','mean'),both_correct_fraction=('both_windows_correct','mean'),conditional_integer_error_rate=('integer_error','mean'),conditional_bias=('error','mean'),inconsistent_edges=('both_windows_correct',lambda v:int((~v).sum())),mu_mean=('mu','mean'),J_mean=('J','mean'),Q_mean=('Q','mean')).reset_index()
        fixed=frame[frame.sigma>0].groupby(keys+['edge_id']).error.var().groupby(keys).mean().rename('fixed_edge_error_variance').reset_index()
        s.merge(fixed,on=keys,how='left').to_csv(path,index=False)
        frame.groupby(['orientation','sigma','window_method','pair_method']).agg(all_geometric_edges=('valid','size'),valid_fraction=('valid','mean'),both_correct_fraction=('both_windows_correct','mean'),conditional_integer_error_rate=('integer_error','mean')).to_csv(path.with_name('overall_'+path.name))
    summarize(p,out/'summary_pairs.csv'); summarize(o,review/'summary_pairs.csv')
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    stats=dict(runtime_seconds=time.perf_counter()-start,window_records=len(w),pair_records=len(p),oracle_records=len(o),strict_success_rate=float(w.strict_success.mean()),failures=int((~w.valid).sum()),integer_error_rate=float(p.integer_error.mean()),max_wrap_error=float(w.wrap_consistency_error.max()))
    (out/'run_stats.json').write_text(json.dumps(stats,indent=2)); print(stats)

if __name__=='__main__': run()
