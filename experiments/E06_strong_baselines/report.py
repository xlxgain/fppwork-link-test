import json,hashlib,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.stats import beta
from run_e06 import ROOT,BASE,history
from e01 import plt

def bootstrap(values,rng,count):
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    if not len(values): return np.nan,np.nan
    means=values[rng.integers(0,len(values),(count,len(values)))].mean(axis=1)
    return np.quantile(means,[.025,.975])

def summary(frame,keys):
    s=frame.groupby(keys).agg(trials=('N_all','size'),N_all=('N_all','sum'),N_out=('N_out','sum'),N_missing=('N_missing','sum'),N_bad_out=('N_bad_out','sum'),continuous_squared_error_sum=('continuous_squared_error_sum','sum')).reset_index()
    s['error_total']=(s.N_bad_out+s.N_missing)/s.N_all; s['covered_error']=s.N_bad_out/s.N_out.replace(0,np.nan); s['coverage']=s.N_out/s.N_all
    s['phase_rmse']=np.sqrt(s.continuous_squared_error_sum/s.N_out.replace(0,np.nan))
    return s

def run():
    out=ROOT/'outputs'; cfg=json.loads((ROOT/'config.json').read_text()); manifest=json.loads((out/'input_manifest.json').read_text()); stats=json.loads((out/'run_stats.json').read_text())
    # Lossless compact representation of phase; discard redundant plotting floats.
    for path in (out/'maps').glob('*_sigma*.npz'):
        with np.load(path) as data:
            payload={k:data[k] for k in data.files if k!='aligned_phase'}
            redundant='aligned_phase' in data.files
        if redundant: np.savez_compressed(path,**payload)
    assert history()==manifest['history_sha256']; assert stats['input_groups']==252 and stats['records']==3528
    assert stats['highest_noise_draws']==40 and stats['total_noise_draws']==280
    t=pd.read_csv(out/'trials.csv'); r=pd.read_csv(out/'roi_trials.csv')
    assert t.groupby(['scene','a_min','sigma','repeat']).size().eq(14).all()
    assert t.groupby(['scene','a_min','sigma','repeat']).highest_input_hash.nunique().eq(1).all()
    assert np.all(t.N_all==t.N_out+t.N_missing)
    sums=r[r.roi.isin(['core','transition','outside'])].groupby(['scene','a_min','sigma','repeat','method'])[['N_all','N_out','N_bad_out','N_missing']].sum()
    check=t.set_index(['scene','a_min','sigma','repeat','method'])
    for col in sums: assert sums[col].sort_index().equals(check[col].sort_index())
    keys=['scene','a_min','sigma','method','frames']; s=summary(t,keys); sr=summary(r,keys+['roi'])
    s.to_csv(out/'summary_all.csv',index=False); sr.to_csv(out/'summary_roi.csv',index=False)
    extra=t.groupby(keys).agg(whole_success=('whole_image_success','sum'),component_success=('component_success','mean'),mean_runtime=('runtime','mean'),absolute_integer_accuracy=('absolute_integer_accuracy','mean'),anchored_accuracy=('single_external_anchor_correct','mean'),mean_circular_rmse=('circular_rmse','mean'),method_failures=('method_failed','sum')).reset_index()
    extra['perfect_trial_probability_low']=extra.apply(lambda x: beta.ppf(.025,x.whole_success,(1 if x.sigma==0 else 20)-x.whole_success+1) if x.whole_success else 0.,axis=1)
    extra['perfect_trial_probability_high']=extra.apply(lambda x: beta.ppf(.975,x.whole_success+1,(1 if x.sigma==0 else 20)-x.whole_success) if x.whole_success<(1 if x.sigma==0 else 20) else 1.,axis=1)
    s=s.merge(extra,on=keys); s.to_csv(out/'summary_all.csv',index=False)
    # Fixed whole-component gauge is already applied in each roi_trial.
    base=r[r.a_min==1][['scene','sigma','repeat','method','roi','N_bad_out']].rename(columns={'N_bad_out':'baseline_bad'})
    propagation=r.merge(base,on=['scene','sigma','repeat','method','roi']); propagation['added_bad']=propagation.N_bad_out-propagation.baseline_bad
    propagation.groupby(keys+['roi']).agg(added_bad=('added_bad','sum'),bad=('N_bad_out','sum'),pixels=('N_all','sum')).to_csv(out/'propagation.csv')
    rng=np.random.default_rng(cfg['bootstrap_seed']); comparisons=[]; levels=[]
    targets=['H2002_skimage',cfg['ablation'],'global_BFS']+list(cfg['protocols'])
    for scene in cfg['scene_indices']:
        for minimum in cfg['a_min']:
            left=t[(t.scene==scene)&(t.a_min==minimum)&(t.sigma==5)&(t.method==cfg['primary'])].sort_values('repeat')
            patchleft=r[(r.scene==scene)&(r.a_min==minimum)&(r.sigma==5)&(r.method==cfg['primary'])&(r.roi=='patch')].sort_values('repeat')
            for ref in targets:
                right=t[(t.scene==scene)&(t.a_min==minimum)&(t.sigma==5)&(t.method==ref)].sort_values('repeat')
                patchright=r[(r.scene==scene)&(r.a_min==minimum)&(r.sigma==5)&(r.method==ref)&(r.roi=='patch')].sort_values('repeat')
                assert len(left)==len(right)==20
                differences={
                    'all_error':left.relative_integer_error_total.to_numpy()-right.relative_integer_error_total.to_numpy(),
                    'patch_error':patchleft.relative_integer_error_total.to_numpy()-patchright.relative_integer_error_total.to_numpy(),
                    'coverage':left.coverage_common.to_numpy()-right.coverage_common.to_numpy()}
                local=[]
                for metric,delta in differences.items():
                    lo,hi=bootstrap(delta,rng,cfg['bootstrap_samples']); margin=cfg['margins']['all' if metric=='all_error' else 'patch' if metric=='patch_error' else 'coverage']
                    row=dict(scene=scene,a_min=minimum,reference=ref,metric=metric,n=20,difference=float(delta.mean()),ci_low=lo,ci_high=hi,margin=margin,within_margin=bool(lo>=-margin and hi<=margin)); comparisons.append(row); local.append(row)
                usable=bool(left.relative_integer_error_total.mean()<=.001 and right.relative_integer_error_total.mean()<=.001 and patchleft.relative_integer_error_total.mean()<=.01 and patchright.relative_integer_error_total.mean()<=.01 and left.coverage_common.mean()>=.99 and right.coverage_common.mean()>=.99)
                ceiling=bool(left.whole_image_success.all() and right.whole_image_success.all())
                if ceiling: status='当前配置共同通过/天花板结果'
                elif all(v['within_margin'] for v in local): status='当前样本中描述性相近' if usable else '共同失败区间'
                else:
                    better=local[0]['ci_high']<0 or local[1]['ci_high']<0 or local[2]['ci_low']>0
                    worse=local[0]['ci_low']>0 or local[1]['ci_low']>0 or local[2]['ci_high']<0
                    status='证据不足（指标取舍）' if better and worse else '主方法错误率更低或覆盖更好' if better else '主方法错误率更高或覆盖更差' if worse else '证据不足'
                levels.append(dict(scene=scene,a_min=minimum,primary=cfg['primary'],reference=ref,status=status,high_accuracy_usable=usable,primary_all_error=left.relative_integer_error_total.mean(),reference_all_error=right.relative_integer_error_total.mean(),primary_patch_error=patchleft.relative_integer_error_total.mean(),reference_patch_error=patchright.relative_integer_error_total.mean(),primary_coverage=left.coverage_common.mean(),reference_coverage=right.coverage_common.mean()))
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False); level=pd.DataFrame(levels); level.to_csv(out/'performance_levels.csv',index=False)
    matchrows=[]
    for (scene,minimum),g in level[level.reference.isin(cfg['protocols'])].groupby(['scene','a_min']):
        matches=g[g.high_accuracy_usable & g.status.isin(['当前配置共同通过/天花板结果','当前样本中描述性相近'])]
        frames=sorted(4*len(cfg['protocols'][name]) for name in matches.reference)
        matchrows.append(dict(scene=scene,a_min=minimum,matched_frame_counts=str(frames),minimum_matching_frames=min(frames) if frames else np.nan,note='仅当前样本/工程容差，非普适等级' if frames else '无高正确率相近协议'))
    matchtable=pd.DataFrame(matchrows); matchtable.to_csv(out/'minimum_matched_frames.csv',index=False)
    shown=['global_BFS','H2002_skimage',cfg['ablation'],cfg['primary']]+list(cfg['protocols'])
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for col,scene in enumerate(cfg['scene_indices']):
        for method in shown:
            for row,roi in [(0,'all'),(1,'patch')]:
                g=sr[(sr.scene==scene)&(sr.sigma==5)&(sr.method==method)&(sr.roi==roi)]
                axs[row,col].plot(g.a_min,g.error_total,'o-',markersize=3,label=method)
                axs[row,col].set(xscale='log',xlabel='Minimum signal coefficient',ylabel=roi+' integer total error',title=scene)
    axs[0,1].legend(fontsize=6); fig.tight_layout(); fig.savefig(out/'signal_vs_integer_error.png',dpi=150); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    for col,scene in enumerate(cfg['scene_indices']):
        g=s[(s.scene==scene)&(s.sigma==5)&(s.method.isin(shown))]
        for ai,minimum in enumerate(cfg['a_min']):
            z=g[g.a_min==minimum]
            for mi,method in enumerate(shown):
                part=z[z.method==method]; marker=['x','s','^','o','D','v','P','*'][mi]
                axs[0,col].scatter(part.frames,1-part.error_total,marker=marker,color=f'C{ai}',s=25,label=method if ai==0 else None)
                axs[1,col].scatter(part.frames,part.coverage,marker=marker,color=f'C{ai}',s=25,label=f'a={minimum}' if mi==0 else None)
        axs[0,col].set(title=scene,ylabel='Correct / common geometry',xlabel='Frames'); axs[1,col].set(ylabel='Output coverage',xlabel='Frames')
    axs[0,1].legend(fontsize=5); axs[1,1].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'frames_vs_performance.png',dpi=150); plt.close(fig)
    pc=pd.DataFrame(comparisons); fig,axs=plt.subplots(1,2,figsize=(11,4))
    for scene,g in pc[(pc.reference=='H2002_skimage')&(pc.metric=='patch_error')].groupby('scene'):
        axs[0].errorbar(g.a_min,g.difference,yerr=np.maximum(0,np.array([g.difference-g.ci_low,g.ci_high-g.difference])),fmt='o-',label=scene)
    axs[0].axhline(0,color='k'); axs[0].set(xscale='log',xlabel='Minimum signal',ylabel='Primary minus H2002 patch error'); axs[0].legend()
    for scene,g in s[(s.method==cfg['primary'])&(s.sigma==5)].groupby('scene'): axs[1].plot(g.a_min,g.phase_rmse,'o-',label=scene)
    axs[1].set(xscale='log',xlabel='Minimum signal',ylabel='Whole-component aligned phase RMSE (rad)'); axs[1].legend(); fig.tight_layout(); fig.savefig(out/'paired_H2002_and_phase_rmse.png',dpi=150); plt.close(fig)
    for scene in cfg['scene_indices']:
        for minimum in [.1,.05]:
            data=np.load(out/'maps'/f'{scene}_a{minimum}_sigma5.npz'); names=data['methods'].tolist(); chosen=['global_BFS','H2002_skimage',cfg['ablation'],cfg['primary'],'TPU3_unit_balanced','TPU5_unit_balanced']
            fig,axs=plt.subplots(3,len(chosen),figsize=(18,8))
            for col,method in enumerate(chosen):
                i=names.index(method); finite=data['finite_masks'][i]; err=np.where(finite,data['relative_integer_errors'][i],np.nan)
                axs[0,col].imshow(err,cmap='coolwarm',vmin=-3,vmax=3); axs[0,col].set_title(method,fontsize=7)
                axs[1,col].imshow(err,cmap='coolwarm',vmin=-3,vmax=3); axs[1,col].set_xlim(75,150); axs[1,col].set_ylim(150,75)
                gauge=data['absolute_integer_errors'][i]-data['relative_integer_errors'][i]
                continuous=np.where(finite,data['highest_phi']+2*np.pi*(data['integer_orders'][i]-gauge)-data['truth'],np.nan); axs[2,col].imshow(continuous,cmap='coolwarm',vmin=-np.pi,vmax=np.pi)
                for row in range(3): axs[row,col].contour(data['zones'],levels=[.5,1.5],colors='k',linewidths=.4)
            fig.suptitle(f'{scene} a={minimum}: full integer / patch view / continuous phase error'); fig.tight_layout(); fig.savefig(out/f'error_maps_{scene}_{minimum}.png',dpi=140); plt.close(fig)
    suites=list(ET.parse(out/'pytest.xml').getroot().iter('testsuite')); tests={v:sum(float(x.get(v,0)) for x in suites) for v in ['tests','failures','errors','skipped','time']}
    stats['pytest']=tests; (out/'run_stats.json').write_text(json.dumps(stats,indent=2))
    manifest['postprocess_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}; (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    primary=s[(s.method==cfg['primary'])&(s.sigma==5)][['scene','a_min','error_total','phase_rmse','coverage','whole_success']]
    coretable=sr[(sr.sigma==5)&(sr.method.isin(['H2002_skimage',cfg['primary']]+list(cfg['protocols'])))&(sr.roi=='patch')][['scene','a_min','method','error_total','coverage','phase_rmse']]
    coretable.to_csv(out/'core_comparison.csv',index=False)
    m0=level[level.reference==cfg['ablation']]
    outside_table=sr[(sr.sigma==5)&(sr.roi=='outside')&sr.method.isin(['global_BFS','H2002_skimage','flat_W0_M0',cfg['ablation'],cfg['primary']])][['scene','a_min','method','N_bad_out','N_missing','error_total']]
    ablation_rows=pd.DataFrame(comparisons); ablation_rows=ablation_rows[ablation_rows.reference==cfg['ablation']]
    m1_identical=bool((ablation_rows.difference==0).all() and (ablation_rows.ci_low==0).all() and (ablation_rows.ci_high==0).all())
    weak_outside_primary=int(outside_table[(outside_table.a_min==.02)&(outside_table.method==cfg['primary'])].N_bad_out.sum())
    weak_outside_h2002=int(outside_table[(outside_table.a_min==.02)&(outside_table.method=='H2002_skimage')].N_bad_out.sum())
    text=f'''# E06 强空间基线与可靠初始化多频对照

基线{manifest['baseline']}；E01–E05逐文件SHA前后相同。全部252组预定输入，3528条整图记录；每个方法每次保留失败，不删除弱档。完整运行{stats['runtime_seconds']:.2f}秒。实际pytest记录{tests}，日志pytest.xml。非零噪声底稿每频40份、七频共280份，最高频只有40份；六档/算法成绩配对相关，不能把像素或3528记录当独立重复数。

## 强基线、频率与公平性

H2002为scikit-image {manifest['H2002']['version']}的二维unwrap_phase，实际签名{manifest['H2002']['signature']}，rng=0、wrap_around=False，masked array屏蔽无效；逐共同连通分量保留零点。输出先核对同余，再表达成原phi加整数周期，不以投影隐藏错误实现。具体论文与库依据见methodology.md。

8/12/16/20帧依次使用[1,60]、[1,8,60]、[1,4,15,60]、[1,3,8,22,60]。最低频在已知投影域和零原点下无噪初始化通过；弱信号时不保证分支正确。所有频率均实际生成四步；算法只读取自身频率，七频池28帧不是每算法帧数。无E05频率2起始歧义基线进入本轮排名。

平面和(0,0)mm光滑凸包保持E05针孔几何，最高f=60。采用E03精确物理斑块中心{manifest['geometry']['plane']['center_XY_mm']}mm、30/50mm半径、六档a。I=(27.5+100a)+100a cos(Φf+kπ/2)+ε，σ=0/5，非零各20次。浮点不裁剪不量化；不引入各频额外MTF，不能推广为真实系统各频同成像。固定背景不是全部环境反射光。参数为预定探索性机制配置。

所有方法共用原几何mask，低调制不提前删除。a=1最高频哈希和E05结果回归通过（仅共同输入分量一个整数参考，不逐窗对齐）。八种六边形组合与E05整数图协调直接调用、原选点和参数不变；主方法预先指定flat-W1-M1。原M1 J/Q低调制下未校准，图仍round(mu)等权L1，不使用新增置信权。

## 评价与原始表

e=round((Psi−T_obs)/2π)，只对输入mask四连通分量各减一个众数；core/transition/outside/patch均沿用此误差图，不独立对齐。总错误=(已输出错+缺失)/全部几何像素；covered-only、覆盖率、相位RMSE独立给出。四帧无锚绝对能力标“整体整数自由度未消除”；原始零点差仅诊断，不用0%给四帧算法作失败排名。外部中心(128,128)锚点附表明确是评分端真值模拟校准，不是观测自身能力。

主方法全图表：
```
{primary.to_string(index=False)}
```
核心patch对照表（全部档位，不删失败）：
```
{coretable.to_string(index=False)}
```
全组合见summary_all.csv、summary_roi.csv；正常区传播见propagation.csv，均按同一噪声重复与a=1配对。完整repeat0相位、全图/patch整数和连续误差见maps与误差图；有限mask独立存储，整数int32无哨兵冲突。其他重复由精确种子与逐频哈希复现。

## 五个问题的直接回答

1. 主方法相对H2002的逐场景/衰减分类如下；依据repeat级2000次配对bootstrap，原始差值与95%区间在paired_comparisons.csv，不能用BFS失效代替与强基线比较。
```
{level[level.reference=='H2002_skimage'].to_string(index=False)}
```
2. 与8/12/16/20帧的等级表见performance_levels.csv。判定同时要求全图±0.001、patch±0.01、覆盖±0.001；高正确率表述额外要求双方全图错误≤0.001、patch≤0.01、覆盖≥0.99。共同明显失效不包装成有效等效。相同零错只叫当前配置共同通过/天花板；20次全成功的试验成功概率双侧精确95%下界约83.16%，逐方法界另存summary_all.csv。
```
{matchtable.to_string(index=False)}
```
若多个协议均饱和，只报告达到同一高正确率相近成绩的最少帧数，不挑最高帧数包装成更高级别。
3. 正常区错误传播与区域化/W1的作用需结合outside计数、W0/W1及全图H2002比较。没有改变区域形状的消融，不单凭当前结果断言六边形形状的独立因果优势；全图接近100%不能替代patch正确。
```
{outside_table.to_string(index=False)}
```
W0与W1同形状、相同图后端对照可描述局部展开选择的增量，但不能把W1拟合与区域支撑的作用统称为“六边形独立优势”。
在a=0.02的两场景20次重复中，主方法outside错误像素合计{weak_outside_primary}个，H2002为{weak_outside_h2002}个；因此不能宣称本方法在所有区域始终更好，也不能用全图平均掩盖弱区失效。a=0.05相对H2002的patch优势与最弱档的证据不足应分别陈述。
4. M1相对M0的完整配对结论如下，任何相同或更差均如实保留；不能把尚未实现的连续置信图后端当成本轮收益。
本轮主方法与M0消融在全图错误、patch错误、覆盖率三个指标的逐条件配对差及区间全部为0：{m1_identical}。若为True，应明确结论是本轮未观察到M1额外收益，不能宣称M1带来改善。
```
{m0.to_string(index=False)}
```
5. 四帧相对8/12/16/20帧分别少50%/66.7%/75%/80%物体帧；代价包括空间连续性、图连通性、绝对锚点、计算开销和弱区可靠性。Python与C实现时间不能直接视为理论速度排名，采集帧减少不等于总耗时同比减少。绝对多频结果独立记录，小数相位噪声对所有方法相同，级次正确也不等于恢复高精度相位。

本轮只验证固定几何/信号机制，未证明双目、远距离、遮挡或任意不连续能力。实拍JPEG未作准确率真值。复现：工作区.venv安装scikit-image==0.25.2，运行`python -m pytest -q . ../E05_frame_equivalence ../E04_real_monocular ../E03_low_modulation ../E02_local_unwrapping ../E01_hex_pairwise --junitxml=outputs/pytest.xml`、`run_e06.py`、`report.py`。实际运行阶段日志run.log，源/依赖版本与历史哈希在input_manifest.json。
'''
    (ROOT/'report_E06.md').write_text(text,encoding='utf-8'); print(coretable.to_string(index=False)); print(stats)

if __name__=='__main__': run()
