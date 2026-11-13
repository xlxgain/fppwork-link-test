import json,sys,time,traceback,hashlib,subprocess,inspect,platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,skimage
ROOT=Path(__file__).resolve().parent; BASE=ROOT.parent
from baselines import FOUR,h2002,hex_outputs,spatial_bfs,hierarchical,unwrap_phase
from observations import prepare,generate,digest
from eval_e06 import evaluate
from algorithms import TAU

def history():
    paths=subprocess.check_output(['git','ls-files','experiments'],cwd=BASE.parent,text=True).splitlines()
    return {p:hashlib.sha256((BASE.parent/p).read_bytes()).hexdigest() for p in paths if not p.startswith('experiments/E06_')}

def run():
    start=time.perf_counter(); out=ROOT/'outputs'; out.mkdir(exist_ok=True); (out/'maps').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'config.json').read_text()); before=history(); rows=[]; rois=[]; inputs=[]; logs=[]; noisedraws={}; regression=[]
    old=json.loads((BASE/'E05_frame_equivalence/outputs/input_manifest.json').read_text()); oldhash={(v['scene'],v['sigma'],v['repeat']):v['frame_hashes'] for v in old['paired_simulation_inputs']}
    expected=FOUR+list(cfg['protocols']); geometry_info={}; source_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    for scene in cfg['scene_indices']:
        z,u,mask,p,center=prepare(scene,cfg); truth=TAU*60*u/1024
        np.savez_compressed(out/'maps'/f'geometry_{scene}.npz',truth=truth,mask=mask,projector_u=u,depth_mm=z)
        geometry_info[scene]=dict(parameters=p,center_XY_mm=center,projector_u_hash=digest(u),mask_hash=digest(mask))
        for minimum in cfg['a_min']:
            for sigma in cfg['sigma']:
                for repeat in range(1 if sigma==0 else cfg['repeats']):
                    prepstart=time.perf_counter(); phases,zones,a,frameinfo=generate(u,z,p,center,scene,minimum,sigma,repeat,cfg); preptime=time.perf_counter()-prepstart
                    common=dict(scene=scene,a_min=minimum,sigma=sigma,repeat=repeat,highest_input_hash=next(v['input_hash'] for v in frameinfo if v['frequency']==60))
                    for v in frameinfo:
                        if sigma: noisedraws.setdefault(str(v['frequency']),set()).add(v['noise_hash'])
                        if minimum==1 and str(v['frequency']) in oldhash[scene,sigma,repeat]: assert v['input_hash']==oldhash[scene,sigma,repeat][str(v['frequency'])]
                    inputs.append(dict(**common,attenuation_hash=digest(a),frequencies=frameinfo))
                    outputs={}; failures={}
                    for orientation in ['flat','pointy']:
                        try:
                            for name,phase,comp,diag,edges in hex_outputs(phases[60],mask,orientation): outputs[name]=(phase,diag)
                        except Exception:
                            reason=traceback.format_exc(); logs.append(reason)
                            for name in FOUR:
                                if name.startswith(orientation) and name not in outputs: failures[name]=reason
                    for name,fn in [('global_BFS',spatial_bfs),('H2002_skimage',h2002)]:
                        try:
                            phase,comp,diag=fn(phases[60],mask); outputs[name]=(phase,diag)
                        except Exception: failures[name]=traceback.format_exc(); logs.append(failures[name])
                    for name,freqs in cfg['protocols'].items():
                        tick=time.perf_counter()
                        try:
                            phase=hierarchical({f:phases[f] for f in freqs},freqs); outputs[name]=(np.where(mask,phase,np.nan),dict(runtime=time.perf_counter()-tick,read_frequencies=str(freqs)))
                        except Exception: failures[name]=traceback.format_exc(); logs.append(failures[name])
                    orders=[]; finite_maps=[]; abs_errors=[]; rel_errors=[]; aligned_maps=[]
                    for name in expected:
                        phase,diag=outputs.get(name,(np.full(mask.shape,np.nan),dict(runtime=np.nan)))
                        score,roi,absolute,residual,aligned=evaluate(phase,phases[60],truth,mask,zones)
                        frames=4 if name in FOUR else 4*len(cfg['protocols'][name])
                        rows.append(dict(**common,method=name,frames=frames,**score,**diag,data_preparation_runtime=preptime,method_failed=name in failures,failure_reason=failures.get(name,''),absolute_capability='global_integer_gauge_unresolved' if name in FOUR else 'unit_frequency_domain_initialized',absolute_integer_accuracy=score['raw_absolute_correct'] if name not in FOUR else np.nan))
                        rois.extend(dict(**common,method=name,frames=frames,**v) for v in roi)
                        finite=mask&np.isfinite(phase)
                        if repeat==0:
                            k=np.zeros(mask.shape,np.int32); k[finite]=np.rint((phase[finite]-phases[60][finite])/TAU).astype(np.int32)
                            assert np.max(abs(k[finite]),initial=0)<2**30
                            orders.append(k); finite_maps.append(finite); abs_errors.append(np.nan_to_num(absolute).astype(np.int32)); rel_errors.append(np.nan_to_num(residual).astype(np.int32)); aligned_maps.append(aligned.astype(np.float32))
                        if minimum==1 and name in FOUR and name!='H2002_skimage':
                            saved=np.load(BASE/'E05_frame_equivalence/outputs/maps'/f'{scene}_sigma{sigma}_repeat{repeat:02d}.npz'); names=saved['methods'].tolist(); i=names.index(name); oldk=saved['integer_orders'][i]; oldfinite=oldk!=-32768
                            assert np.array_equal(finite,oldfinite)
                            delta=np.rint((phase[finite]-saved['highest_phi'][finite])/TAU)-oldk[finite]
                            assert len(np.unique(delta))<=1 # common mask is one component; NEVER each window
                            regression.append(dict(**common,method=name,passed=True))
                    if repeat==0:
                        np.savez_compressed(out/'maps'/f'{scene}_a{minimum}_sigma{sigma}.npz',methods=np.array(expected),highest_phi=phases[60],integer_orders=np.stack(orders),finite_masks=np.stack(finite_maps),absolute_integer_errors=np.stack(abs_errors),relative_integer_errors=np.stack(rel_errors),zones=zones,truth=truth)
                log=f'{scene} a={minimum} sigma={sigma}: complete'; logs.append(log); print(log,flush=True)
                pd.DataFrame(rows).to_csv(out/'trials.csv',index=False); pd.DataFrame(rois).to_csv(out/'roi_trials.csv',index=False)
    assert len(rows)==3528 and history()==before
    manifest=dict(baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,geometry=geometry_info,inputs=inputs,independent_noise_draws_per_frequency={f:len(v) for f,v in noisedraws.items()},H2002=dict(version=skimage.__version__,signature=str(inspect.signature(unwrap_phase)),rng=0,wrap_around=False),environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,pandas=pd.__version__,platform=platform.platform()),source_hashes=source_hash,history_sha256=before,history_unchanged=True)
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); pd.DataFrame(regression).to_csv(out/'E05_regression.csv',index=False)
    stats=dict(runtime_seconds=time.perf_counter()-start,input_groups=len(inputs),records=len(rows),method_failures=sum(v['method_failed'] for v in rows),highest_noise_draws=len(noisedraws['60']),total_noise_draws=sum(len(v) for v in noisedraws.values()))
    (out/'run_stats.json').write_text(json.dumps(stats,indent=2)); (out/'run.log').write_text('\n'.join(logs),encoding='utf-8'); print(stats)

if __name__=='__main__': run()
