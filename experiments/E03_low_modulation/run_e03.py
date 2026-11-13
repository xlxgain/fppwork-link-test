import sys,json,time,hashlib,platform,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
E02=ROOT.parent/'E02_local_unwrapping'; E01=ROOT.parent/'E01_hex_pairwise'
sys.path.insert(0,str(E01)); sys.path.insert(0,str(E02))
from e01 import layout,estimate,demodulate,plt
from e01_c import support
from local_unwrap import unwrap_window,wrap,TAU
from radiometry import attenuation,noise_field,intensities,amplitude,digest
from evaluation_e03 import evaluate,reference,observation,propagation

def historical_hashes():
    return {str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [E01,E02] for p in folder.rglob('*') if p.is_file() and not any(v in p.parts for v in ['.git','__pycache__','.pytest_cache','reference_tmp']) and p.name!='reference_tree.json'}

def run():
    tick=time.perf_counter(); cfg=json.loads((ROOT/'config.json').read_text()); out=ROOT/'outputs'; out.mkdir(exist_ok=True)
    shards=out/'records'; shards.mkdir(exist_ok=True)
    old=json.loads((E02/'outputs_E02/input_manifest.json').read_text()); intrinsics=old['geometry']; center=old['scenes']['edge']['bump_center_XY_mm']
    manifest=dict(baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),config=cfg,center_XY_mm=center,geometry=intrinsics,history_before=historical_hashes(),source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},inputs=[],files=[],geometries={})
    observations=[]; y,x=np.indices((256,256)); coords=np.stack([x,y],-1)
    e02_hash={(r['scene_id'],r['sigma'],r['repeat']):r['input_hash'] for r in old['inputs']}
    for scene,source_index in cfg['source_scene_indices'].items():
        source=E02/'outputs_C_review'/f'{scene}.npz'; data=np.load(source); truth=data['phase_radians']; mask=data['valid']
        # Frozen E02 arrays must match the original E01-C flat arrays exactly.
        original=np.load(E01/'outputs_C'/f'flat_{scene}.npz')
        for field in data.files: assert np.array_equal(data[field],original[field],equal_nan=True)
        manifest['geometries'][scene]=dict(file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),phase_hash=digest(truth),mask_hash=digest(mask),source_scene_index=source_index)
        grids={}
        for orientation in ['flat','pointy']:
            labels,c,inc,edges=layout(orientation=orientation)
            grids[orientation]=(labels,c,inc,support(labels,c,edges,mask,inc))
        for minimum in cfg['a_min']:
            a,zones=attenuation(data['depth_mm'],intrinsics,center,minimum); Btrue=100*a
            for sigma in cfg['sigma']:
                for repeat in range(1 if sigma==0 else cfg['repeats']):
                    noise=noise_field(truth.shape,cfg['seed'],source_index,sigma,repeat)
                    images=intensities(truth,a,noise); input_hash=digest(images)
                    if minimum==1: assert input_hash==e02_hash[(scene,sigma,repeat)]
                    phi=demodulate(images); T=truth+wrap(phi-wrap(truth)); Bhat=amplitude(images)
                    if sigma==0: assert np.max(abs(wrap(phi[mask]-truth[mask])))<1e-12
                    common=dict(scene_id=scene,a_min=minimum,sigma=sigma,repeat=repeat,input_hash=input_hash,noise_draw_id=f'{scene}_{sigma}_{repeat}',attenuation_hash=digest(a))
                    manifest['inputs'].append(dict(**common,noise_hash=digest(noise),intensity_min=float(np.nanmin(images)),intensity_max=float(np.nanmax(images)),outside_0_255_count=int(np.sum((images<0)|(images>255)))))
                    observations.extend(dict(**common,**v) for v in observation(phi,truth,Bhat,Btrue,mask,zones))
                    wr=[]; pr=[]; oracle=[]
                    for orientation,(labels,c,inc,es) in grids.items():
                        for wm in ['W0','W1_MLSQU_HEX_Q2','oracle_control']:
                            raw=np.full(phi.shape,np.nan); scores={}; diags={}; residualmap=np.full(phi.shape,np.nan)
                            for i in range(len(c)):
                                region=(labels==i)&mask
                                if wm=='oracle_control':
                                    m=np.random.default_rng(old['config']['reference_seed']+i).integers(-8,9)
                                    psi=np.where(region,T-TAU*m,np.nan); diag=dict(valid=bool(region.any()),reason='',n_components=1,model_rank=0,condition=np.nan,rho_grid_index=-1,runtime=0.)
                                else: psi,diag=unwrap_window(phi,region,coords,dict(method=wm,center=c[i]))
                                score,residual=evaluate(psi,phi,truth,region,diag,zones)
                                raw[region]=psi[region]; residualmap[region]=residual[region]; scores[i]=score; diags[i]=diag
                                if wm!='oracle_control': wr.append(dict(**common,orientation=orientation,window_method=wm,window_id=i,connection='truncated' if inc[i] else 'internal',**diag,**score))
                            if sigma==5 and repeat==0 and wm!='oracle_control':
                                # Few small diagnostic maps per predeclared attenuation, no intensity stacks.
                                np.savez_compressed(out/f'map_{scene}_{minimum}_{orientation}_{wm}.npz',residual_integer=residualmap.astype(np.float32))
                                if minimum==.05 and scene=='edge' and orientation=='flat':
                                    np.savez_compressed(out/f'example_{wm}.npz',phi=phi,psi=raw,B_true=Btrue,B_hat=Bhat,zones=zones,frame0=images[0],circular_error=wrap(phi-truth))
                            for edge_id,e in enumerate(es):
                                begin=time.perf_counter(); i,j=e['i'],e['j']; ref=reference(scores[i],scores[j]); inputs_valid=diags[i]['valid'] and diags[j]['valid']
                                values=estimate(raw,phi,e,sigma*sigma/20000 if sigma else 1) if len(e['pairs']) and inputs_valid else [(m,np.nan,np.nan,np.nan,False,0.) for m in ['M0','M1']]
                                variance,group=propagation(e,Btrue,sigma)
                                affected=bool(np.any(zones.ravel()[e['ix']]!=2)) if len(e['ix']) else False
                                for pm,mu,J,Q,valid,retained in values:
                                    unique=ref['both_windows_correct']; d=ref['d_reference']; integer_error=float(round(mu)!=d) if unique and valid else np.nan
                                    row=dict(**common,orientation=orientation,window_method=wm,pair_method=pm,i=i,j=j,edge_id=edge_id,direction=e['direction'],connection=e['connection'],patch_related=affected,mu=mu,J_legacy_homoscedastic=J,Q_legacy_homoscedastic=Q,uncertainty_status='unit_geometry_only' if sigma==0 else 'legacy_uniform_variance_uncalibrated',v_mu_firstorder_oracle=variance if pm=='M1' else np.nan,snr_diagnostic_group=group,**ref,error=mu-d if unique and valid else np.nan,integer_error=integer_error,end_to_end_success=bool(unique and valid and integer_error==0),valid=valid,invalid_reason='' if valid else 'window_failure' if not inputs_valid else 'empty_support' if not len(e['pairs']) else 'rank_or_condition_failure',support_pixel_count=len(e['ix']),crossing_pair_count=len(e['pairs']),rank=e['rank'],condition=e['condition'],retained_fraction=retained,runtime=time.perf_counter()-begin)
                                    (oracle if wm=='oracle_control' else pr).append(row)
                    stem=f'{scene}_a{minimum}_s{sigma}_r{repeat:02d}'
                    for kind,rows in [('windows',wr),('pairs',pr),('oracle_pairs',oracle)]:
                        filename=shards/f'{kind}_{stem}.csv'; pd.DataFrame(rows).to_csv(filename,index=False)
                        manifest['files'].append(dict(kind=kind,path=str(filename.relative_to(out)),rows=len(rows),sha256=hashlib.sha256(filename.read_bytes()).hexdigest()))
                print(f'{scene} a={minimum} sigma={sigma} complete',flush=True)
    pd.DataFrame(observations).to_csv(out/'observation_summary.csv',index=False)
    assert manifest['history_before']==historical_hashes()
    manifest['history_unchanged']=True; manifest['independent_nonzero_noise_draws']=len({r['noise_hash'] for r in manifest['inputs'] if r['sigma']>0})
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    stats=dict(runtime_seconds=time.perf_counter()-tick,input_groups=len(manifest['inputs']),noise_draws=manifest['independent_nonzero_noise_draws'],history_unchanged=True)
    (out/'run_stats.json').write_text(json.dumps(stats,indent=2)); print(stats)

if __name__=='__main__': run()
