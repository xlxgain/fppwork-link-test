import sys,json,time,hashlib,subprocess,platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parent; BASE=ROOT.parent
sys.path.insert(0,str(BASE/'E01_hex_pairwise'))
from e01_c import geometry
from e01 import demodulate,plt
from algorithms import hex_outputs,spatial_bfs,hierarchical,wrap,TAU
from evaluation_e05 import evaluate,mode

def history():
    return {str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for name in ['E01_hex_pairwise','E02_local_unwrapping','E03_low_modulation','E04_real_monocular'] for p in (BASE/name).rglob('*') if p.is_file() and not any(v in p.parts for v in ['__pycache__','.pytest_cache','data_cache','reference_tmp']) and p.name not in ['inspection.png','reference_tree.json']}

def simulated_geometry(scene,p,step):
    if scene!='depth_step':
        _,mask,z,u,v=geometry(p,None if scene=='plane' else (0,0)); return z,u,mask
    y,x=np.indices(p['shape']); rx=(x-p['cx'])/p['fc']; ry=(y-p['cy'])/p['fc']
    z=(p['distance_mm']-step*(rx>=0))/(1-p['tilt_x']*rx-p['tilt_y']*ry)
    u=p['fp']*(z*rx-p['baseline_mm'])/z+p['pcx']; v=p['fp']*ry+p['pcy']
    mask=(z>0)&(u>=0)&(u<p['projector_shape'][1])&(v>=0)&(v<p['projector_shape'][0])
    return z,u,mask

def simulated_observations(u,cfg,scene_index,sigma,repeat):
    phases={}; hashes={}; high=None
    for f in cfg['frequencies']:
        truth=TAU*f*u/1024
        images=cfg['A']+cfg['B']*np.cos(truth[None]+np.arange(4)[:,None,None]*np.pi/2)
        images+=np.random.default_rng(np.random.SeedSequence([cfg['seed'],scene_index,int(sigma),repeat,f])).normal(0,sigma,images.shape)
        phases[f]=demodulate(images); hashes[f]=hashlib.sha256(images.tobytes()).hexdigest()
    return phases,hashes

def methods(phases,mask,cfg,include_five=True):
    high=phases[60]
    # Four-frame outputs produced without access to lower frequencies.
    for ori in cfg['orientation']:
        for name,phase,components,diag,edges in hex_outputs(high,mask,ori):
            yield name,4,phase,components,diag,edges
    phase,components,diag=spatial_bfs(high,mask); yield 'global_BFS',4,phase,components,diag,[]
    for name,freqs in cfg['protocols'].items():
        if name=='TPU5' and not include_five: continue
        start=time.perf_counter(); phase=np.where(mask,hierarchical(phases,freqs),np.nan)
        components,_=label(mask)
        yield name,4*len(freqs),phase,components,dict(runtime=time.perf_counter()-start,initialization='positive_principal_lowest; unverified for frequency 2'),[]

def pack_output(target,names,phi,phases,abs_errors=None,relative_errors=None):
    # Lossless representation: full phase = highest_phi + 2pi*integer_orders.
    k=np.stack([np.where(np.isfinite(p),np.rint((np.nan_to_num(p)-phi)/TAU),-32768).astype(np.int16) for p in phases])
    payload=dict(methods=np.array(names),highest_phi=phi,integer_orders=k,missing_sentinel=-32768)
    if abs_errors is not None:
        payload.update(absolute_integer_errors=np.stack([np.where(np.isfinite(a),a,-32768).astype(np.int16) for a in abs_errors]),relative_integer_errors=np.stack([np.where(np.isfinite(a),a,-32768).astype(np.int16) for a in relative_errors]))
    np.savez_compressed(target,**payload)

def run():
    start=time.perf_counter(); cfg=json.loads((ROOT/'config.json').read_text()); out=ROOT/'outputs'; out.mkdir(exist_ok=True); (out/'maps').mkdir(exist_ok=True)
    histories=history(); baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    p=json.loads((BASE/'E01_hex_pairwise/parameters_C.json').read_text()); records=[]; inputs=[]; runtimes=[]
    for si,scene in enumerate(cfg['scenes']):
        z,u,mask=simulated_geometry(scene,p,cfg['step_mm']); truth=TAU*60*u/1024
        anchor=int(np.argmin(np.where(mask,(np.indices(mask.shape)[0]-128)**2+(np.indices(mask.shape)[1]-128)**2,np.inf)))
        np.savez_compressed(out/'maps'/f'geometry_{scene}.npz',truth=truth,depth_mm=z,projector_u=u,mask=mask,anchor_index=anchor)
        for sigma in cfg['sigma_I']:
            for repeat in range(1 if sigma==0 else cfg['noisy_repeats']):
                phases,hashes=simulated_observations(u,cfg,si,sigma,repeat); inputs.append(dict(scene=scene,sigma=sigma,repeat=repeat,frame_hashes=hashes))
                names=[]; allphase=[]; ae=[]; re=[]; edgerows=[]
                for name,frames,phase,components,diag,edges in methods(phases,mask,cfg):
                    score,a,b=evaluate(phase,phases[60],truth,mask,anchor)
                    records.append(dict(scene=scene,sigma=sigma,repeat=repeat,method=name,frames=frames,**score,**diag))
                    names.append(name); allphase.append(phase); ae.append(a); re.append(b)
                    if repeat==0 and sigma==5: edgerows.extend(dict(method=name,**e) for e in edges)
                pack_output(out/'maps'/f'{scene}_sigma{sigma}_repeat{repeat:02d}.npz',names,phases[60],allphase,ae,re)
                if edgerows: pd.DataFrame(edgerows).to_csv(out/f'edges_{scene}.csv',index=False)
            print('simulation',scene,sigma,'complete',flush=True)
    df=pd.DataFrame(records); df.to_csv(out/'sim_trials.csv',index=False)
    # Real source: exactly same highest four frames for every method.
    e04=BASE/'E04_real_monocular'; realcfg=json.loads((e04/'config.json').read_text()); cache=e04/'data_cache'
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cache,text=True).strip(); assert source==realcfg['source_commit']
    from PIL import Image
    realphases={}; diffphases={}; realhashes={}; mods=[]; dcs=[]
    for k,f in enumerate([2,4,14,60]):
        for kind,groups in [('object',realcfg['object_groups']),('reference',realcfg['reference_groups'])]:
            images=np.array([np.asarray(Image.open(cache/'data/sphere'/f'DSC_{i:04d}.JPG').convert('L'),dtype=float) for i in groups[k]])
            realhashes[f'{kind}_{f}']=hashlib.sha256(images.tobytes()).hexdigest()
            phi=demodulate(images)
            if kind=='object':
                realphases[f]=phi; mods.append(.5*np.hypot(images[0]-images[2],images[3]-images[1])); dcs.append(images.mean(axis=0))
            else: diffphases[f]=wrap(realphases[f]-phi)
    # All methods share HIGHEST mask, not a success-dependent intersection.
    mask=(mods[-1]>.1*np.percentile(mods[-1],99))&(dcs[-1]>15)&(dcs[-1]<250)
    names=[]; outputs=[]; realrows=[]; realcomponents=[]
    for name,frames,phase,components,diag,edges in methods(realphases,mask,cfg,False):
        finite=mask&np.isfinite(phase)
        realrows.append(dict(method=name,frames=frames,coverage_common=float(finite.sum()/mask.sum()),coverage_all=float(finite.mean()),absolute_integer_accuracy=np.nan,reference_status='no_trusted_absolute_reference',closure=float(np.max(abs(wrap(phase[finite]-realphases[60][finite])))),**diag))
        names.append(name); outputs.append(phase); realcomponents.append(components)
        print('real',name,'complete',flush=True)
    pack_output(out/'maps'/'real_full_outputs.npz',names,realphases[60],outputs)
    # E04 source-style difference TPU is distinct from the primary ABSOLUTE input.
    deltarows=[]
    for name,freqs in cfg['protocols'].items():
        if name=='TPU5': continue
        d=hierarchical(diffphases,freqs); deltarows.append(dict(method=name,definition='object minus reference',phase_min=float(d[mask].min()),phase_max=float(d[mask].max()),absolute_status='unverified_lowest_order_and_origins'))
    pd.DataFrame(deltarows).to_csv(out/'real_difference_protocol_checks.csv',index=False)
    disagreements=[]; comp,ncomp=label(mask)
    refidx=names.index('TPU4'); reference=outputs[refidx]
    for name,phase in zip(names,outputs):
        finite=mask&np.isfinite(phase)&np.isfinite(reference); orders=np.rint((phase-reference)/TAU)
        aligned=orders.copy()
        for c in range(1,ncomp+1):
            ok=finite&(comp==c)
            if ok.any(): aligned[ok]-=mode(orders[ok].astype(int))
        disagreements.append(dict(method=name,paired_pixels=int(finite.sum()),raw_disagreement=float(np.mean(orders[finite]!=0)),component_relative_disagreement=float(np.mean(aligned[finite]!=0)),interpretation='agreement only, TPU4 is NOT truth'))
    pd.DataFrame(disagreements).to_csv(out/'real_disagreement.csv',index=False); pd.DataFrame(realrows).to_csv(out/'real_methods.csv',index=False)
    manifest=dict(baseline=baseline,config=cfg,simulation_geometry=p,real_source=realcfg['source'],real_commit=source,real_input_hashes=realhashes,paired_simulation_inputs=inputs,environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),historical_files_unchanged=histories==history(),historical_sha256=histories,source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})
    assert manifest['historical_files_unchanged']
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    stats=dict(runtime_seconds=time.perf_counter()-start,simulation_trials=63,simulation_records=len(df),real_methods=len(names),highest_frequency=60,history_unchanged=True)
    (out/'run_stats.json').write_text(json.dumps(stats,indent=2)); print(stats)

if __name__=='__main__': run()
