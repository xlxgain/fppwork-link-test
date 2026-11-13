import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
from run_e05 import ROOT,BASE,plt,history

def interval(values,rng,draws=1000):
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    if not len(values): return np.nan,np.nan
    samples=values[rng.integers(0,len(values),(draws,len(values)))].mean(axis=1)
    return tuple(np.quantile(samples,[.025,.975]))

def run():
    out=ROOT/'outputs'; cfg=json.loads((ROOT/'config.json').read_text()); stats=json.loads((out/'run_stats.json').read_text()); manifest=json.loads((out/'input_manifest.json').read_text())
    assert history()==manifest['historical_sha256']
    manifest['postprocessing_source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    d=pd.read_csv(out/'sim_trials.csv'); rng=np.random.default_rng(17); rows=[]
    for (scene,sigma,method,frames),g in d.groupby(['scene','sigma','method','frames']):
        row=dict(scene=scene,sigma=sigma,method=method,frames=frames,repeats=len(g),N_common=int(g.N_common.sum()),N_out=int(g.N_out.sum()),relative_accuracy=float((g.N_out-g.relative_bad).sum()/g.N_common.sum()),absolute_accuracy=float((g.N_out-g.absolute_bad).sum()/g.N_common.sum()),coverage_common=float(g.N_out.sum()/g.N_common.sum()),coverage_all=float(g.coverage_all.mean()),component_success=float(g.relative_component_success.mean()),relative_rmse=float(np.sqrt(np.mean(g.relative_rmse**2))),absolute_rmse=float(np.sqrt(np.mean(g.absolute_rmse**2))),fractional_rmse=float(np.sqrt(np.mean(g.fractional_phase_rmse**2))),anchored_accuracy=float(g.anchored_correct_all_common.mean()),runtime_mean=float(g.runtime.mean()))
        row['relative_ci_low'],row['relative_ci_high']=interval(g.relative_correct_all_common,rng)
        complete=int(np.sum(g.relative_component_success==1)); n=len(g)
        row['perfect_trial_count']=complete
        row['perfect_trial_probability_exact95_low']=float(beta.ppf(.025,complete,n-complete+1)) if complete else 0.
        row['perfect_trial_probability_exact95_high']=float(beta.ppf(.975,complete+1,n-complete)) if complete<n else 1.
        rows.append(row)
    s=pd.DataFrame(rows); s.to_csv(out/'summary_simulation.csv',index=False)
    paired=[]
    for scene,g in d[d.sigma==5].groupby('scene'):
        for method in g[g.frames==4].method.unique():
            for reference in cfg['protocols']:
                left=g[g.method==method]; right=g[g.method==reference]
                m=left.merge(right,on='repeat',suffixes=('_four','_temporal'))
                for metric in ['relative_correct_all_common','absolute_correct_all_common','anchored_correct_all_common','coverage_all']:
                    delta=m[metric+'_four']-m[metric+'_temporal']; lo,hi=interval(delta,rng,cfg['bootstrap_samples'])
                    paired.append(dict(scene=scene,four_method=method,temporal_method=reference,metric=metric,n=len(delta),difference=float(delta.mean()),ci_low=lo,ci_high=hi,within_predeclared_one_percent=bool(lo>=-cfg['equivalence_margin_fraction'] and hi<=cfg['equivalence_margin_fraction']),temporal_initialization='unit_frequency_calibrated_domain' if reference=='TPU5' else 'frequency2_global_initialization_ambiguous'))
    pairs=pd.DataFrame(paired); pairs.to_csv(out/'paired_equivalence.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    for ax,scene in zip(axs,cfg['scenes']):
        g=s[(s.scene==scene)&(s.sigma==5)]
        for method,part in g.groupby('method'):
            ax.errorbar(part.frames,part.relative_accuracy,yerr=[part.relative_accuracy-part.relative_ci_low,part.relative_ci_high-part.relative_accuracy],fmt='o',label=method,markersize=4)
        ax.set(title=scene,xlabel='Captured object frames',ylabel='Relative correct / common valid pixels',xticks=[4,8,12,16,20],ylim=(-.05,1.05))
    axs[-1].legend(fontsize=5,loc='lower right'); fig.tight_layout(); fig.savefig(out/'frames_vs_relative_accuracy.png',dpi=150); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    for ax,scene in zip(axs,cfg['scenes']):
        g=s[(s.scene==scene)&(s.sigma==5)]
        for method,part in g.groupby('method'): ax.plot(part.frames,part.coverage_all,'o',label=method,markersize=4)
        ax.set(title=scene,xlabel='Captured object frames',ylabel='Finite outputs / entire image',xticks=[4,8,12,16,20],ylim=(-.05,1.05))
    axs[-1].legend(fontsize=5); fig.tight_layout(); fig.savefig(out/'frames_vs_coverage.png',dpi=150); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for method,g in s[s.sigma==5].groupby('method'):
        # Equal-size frozen scenes; this panel is descriptive pooled means.
        frames=int(g.frames.iloc[0]); axs[0].plot(frames,g.absolute_accuracy.mean(),'o',label=method)
        axs[1].plot(frames,g.anchored_accuracy.mean(),'o')
    axs[0].set(title='Absolute: NO post-alignment',ylabel='Correct / common pixels'); axs[1].set(title='One EXTERNAL exact simulation anchor')
    for ax in axs: ax.set(xlabel='Object frames (calibration cost excluded)',xticks=[4,8,12,16,20],ylim=(-.05,1.05))
    axs[0].legend(fontsize=5); fig.suptitle('Frequency-2 TPU initialization is ambiguous; not a certified traditional benchmark',fontsize=9); fig.tight_layout(); fig.savefig(out/'frames_vs_absolute_accuracy.png',dpi=150); plt.close(fig)
    for scene in cfg['scenes']:
        data=np.load(out/'maps'/f'{scene}_sigma5_repeat00.npz'); names=data['methods'].tolist(); selected=['flat_W1_MLSQU_HEX_Q2_M0','flat_W1_MLSQU_HEX_Q2_M1','global_BFS','TPU2','TPU3','TPU4','TPU5']
        fig,axs=plt.subplots(3,len(selected),figsize=(21,8))
        for col,name in enumerate(selected):
            i=names.index(name); k=data['integer_orders'][i]; phase=np.where(k!=-32768,data['highest_phi']+2*np.pi*k,np.nan)
            axs[0,col].imshow(phase); axs[0,col].set_title(name,fontsize=7)
            for row,field in [(1,'relative_integer_errors'),(2,'absolute_integer_errors')]:
                a=data[field][i].astype(float); a[a==-32768]=np.nan; axs[row,col].imshow(a,cmap='coolwarm',vmin=-30,vmax=30)
            for row in range(3): axs[row,col].axis('off')
        fig.suptitle(scene+': phase / relative integer error / absolute integer error'); fig.tight_layout(); fig.savefig(out/f'phase_error_{scene}.png',dpi=130); plt.close(fig)
    real=np.load(out/'maps'/'real_full_outputs.npz'); names=real['methods'].tolist(); reference=names.index('TPU4')
    selected=['flat_W1_MLSQU_HEX_Q2_M0','flat_W1_MLSQU_HEX_Q2_M1','global_BFS','TPU2','TPU3','TPU4']
    fig,axs=plt.subplots(2,len(selected),figsize=(18,7))
    for col,name in enumerate(selected):
        i=names.index(name); k=real['integer_orders'][i]; valid=(k!=-32768)&(real['integer_orders'][reference]!=-32768)
        phase=np.where(k!=-32768,real['highest_phi']+2*np.pi*k,np.nan)
        diff=np.where(valid,k.astype(float)-real['integer_orders'][reference],np.nan)
        axs[0,col].imshow(phase); axs[0,col].set_title(name,fontsize=7); axs[1,col].imshow(diff,cmap='coolwarm',vmin=-30,vmax=30)
        for row in range(2): axs[row,col].axis('off')
    fig.suptitle('Real full phase / disagreement against UNVERIFIED TPU4 (not truth)'); fig.tight_layout(); fig.savefig(out/'real_disagreement_maps.png',dpi=130); plt.close(fig)
    suites=list(ET.parse(out/'pytest.xml').getroot().iter('testsuite')); tests={key:sum(float(x.get(key,0)) for x in suites) for key in ['tests','failures','errors','skipped','time']}
    stats['pytest']=tests; (out/'run_stats.json').write_text(json.dumps(stats,indent=2))
    compact=s[s.sigma==5][['scene','method','frames','relative_accuracy','absolute_accuracy','anchored_accuracy','coverage_all','component_success','relative_rmse','runtime_mean']]
    validcomparison=pairs[(pairs.temporal_method=='TPU5')&(pairs.metric=='relative_correct_all_common')]
    text=f'''# E05 四帧单频与多频层级基准

## 基线与实际执行

仓库基线{manifest['baseline']}，对应已核对远程E04分支；本轮experiment/E05-frame-equivalence。历史E01–E04逐文件哈希未变。实际pytest {tests}，日志outputs/pytest.xml；完整仿真及真实数据处理{stats['runtime_seconds']:.2f}秒。3场景×(无噪1次+噪声20次)=63组配对观测，{stats['simulation_records']}条方法记录。每个观测保存所有方法完整相位和逐像素整数误差，见outputs/maps：phase=highest_phi+2πinteger_orders，缺失−32768。最高相位phi保存float64，阶次无损整数压缩，不是缩略图。

## 协议、频率与可识别性

方法实现和文献核查见methodology.md。四帧法=W0/W1原局部输出+M0/M1原边测量+等权L1整数图协调；HiGHS双单纯形整数性检查。图分量零点只由最小节点定0，不读真值或任何低频结果。global_BFS为E02 W0在每个输入mask连通分量的简单全图基线。最高四帧在每个试验中完全共享，所有输出对同一个phi60同余，图协调只加整数周期。单窗、选点、图权重均未通过参数搜索优化。

8帧[2,60]、12帧[2,14,60]、16帧[2,4,14,60]为标准层级预测取整；不是最优外差，也不是数论/CRT。频率2覆盖整个投影域时不能唯一确定最低级次。它们的正主值初始化只是一项明确记录的尝试，未用真值修复。**不能把其初始化失败当作四帧优于可靠初始化的传统算法。** 仿真真正生成第五频率1，20帧[1,2,4,14,60]在已知u_p∈[0,1024)及相位零原点条件下可初始化。5帧双频、7帧三频只完成文献协议核查，因缺少合法采集协议/共享光度量及最高频相同四步观测约束，本轮不赋予成绩，作为后续基线。

仿真沿用E01针孔内外参与工作距离；最高频率60定义为投影仪全宽1024上的周期数，周期1024/60像素。平面、12mm/25mm光滑凸包、X=0处40mm两半平面深度台阶，均预先固定；不模拟遮挡或立面阴影。这个台阶不保证产生超过π的采样跳变，不能把其成功解释成任意大不连续恢复能力。各频独立四步强度A=127.5、B=100，σ=0/5，不改变最终相位。所有方法共用最高频有效域，另给全图覆盖率。

## 相对、绝对与锚点评价

相位小数误差wrap(phi60−truth)对全部方法相同。整数误差以同余含噪参考T_obs定义，仅评价器可见。相对评价允许输入mask的每个四连通分量一个整数，**不允许按算法断开的图分量分别对齐掩盖失败**；绝对评价不对齐。附加anchored列只给一个固定中心像素的外部级次，使用仿真真值模拟校准，所有方法同样处理；不把它计作四帧算法自身能力，未逐图分量分别锚定。没有外部锚点，四帧整体整数不可辨识；常数级次正确RMSE不能当作绝对正确。

## 实际噪声对照结果
```
{compact.to_string(index=False)}
```
表中覆盖率分母为全部图像；正确率分母为统一有效像素，缺失算失败。分量成功要求该输入分量完整覆盖且整数残差全零。sigma=0结果、共同小数误差RMSE与绝对RMSE另见summary_simulation.csv；图协调边残差在edges_*.csv。运行时间包含实现开销，不作为经过优化的速度排名。

## 等效性能可以与不可以怎样表述

逐场景、逐四帧组合与各多频方案的配对差、95% bootstrap区间见paired_equivalence.csv；预定±1个百分点界作为本轮描述性等效标准，并不是对所有物体/噪声的保证。帧数—相对正确率与帧数—全图覆盖率见对应两图。20次重复在零失败时的退化bootstrap区间不能证明总体错误概率为零。

本次实际三种场景中，四帧所有组合与全图BFS的相对整数正确率、分量成功率和覆盖率均为100%，与具有单位最低频先验的20帧TPU5相近；配对等效界检查为{bool(validcomparison.within_predeclared_one_percent.all())}。无锚点四帧绝对级次正确率为0，是选定算法零点与物理绝对零点不同的结果，不是证明其整体整数可辨识；一处仿真外部锚定后恢复100%。2起始的8/12/16帧受初始化歧义影响约有30%像素错级次，因此这些结果只说明该初始化不可靠，不能作为可靠传统基线被四帧击败的证据。20次全成功的试验成功概率双侧精确95%下界约83.16%，相应逐方法界已保存在summary_simulation.csv；不能据此认定实际系统必然零错误。

四帧法依赖窗内相对关系、邻边可测以及图连接能提供足够约束；绝对级次还需要外部锚点。TPU5依赖单位最低频、精确频率比例和已知投影相位范围，原则上不要求物体空间连续。对于本轮满足条件且相对准确率相近的场景，可以称为“相对级次指标相近”；不能自动称为绝对相位等效。若四帧图断连或局部内部错，一个锚点与整数图都不能凭空恢复；本轮观测界限需以实际表格为准，不扩展到未测试困难场景。

相对8/12/16/20帧，4帧分别减少4/8/12/16帧，即50%/66.7%/75%/80%。这些节省不包含参考板标定/锚点采集成本，可靠性与先验不能隐去。本轮对2起始基线缺少全域可靠起点，因此不能给出“4帧普遍等效N频”的单一排名；20帧基线和锚点附加结果提供可解释的比较。

## 真实球面数据

复用E04固定外部版本{manifest['real_commit']}的1152×768 JPEG，没有额外下采样。球面频率[2,4,14,60]由源脚本与FFT粗核对，精确频率标定、最低级次和相位原点仍不充分。主比较是物体绝对phi；另单独检查物体−参考差分协议，两者不混用。实拍仍没有可信绝对整数真值，不运行不存在的五频，不将TPU4作为真值。
```
{pd.read_csv(out/'real_methods.csv')[['method','frames','coverage_common','coverage_all','absolute_integer_accuracy']].to_string(index=False)}
```
方法间原始与按共同输入分量对齐的一致性见real_disagreement.csv；分歧图标明参照未验证TPU4。相位闭合仅证明与同一四帧观测一致，不能自证真实级次。

## 复现与交付

在本目录使用工作区.venv执行`python -m pytest -q . ../E04_real_monocular ../E03_low_modulation ../E02_local_unwrapping ../E01_hex_pairwise --junitxml=outputs/pytest.xml`、`run_e05.py`、`report.py`。真实数据缓存依赖E04的fetch_data.py获取固定外部版本，未重复提交JPEG。NPZ含所有试验完整相位及仿真误差图；统计、来源哈希、固定配置和文献限定一并保存。本轮不提出新的单窗检测器，也没有依据结果追加参数档位。
'''
    (ROOT/'report_E05.md').write_text(text,encoding='utf-8'); print(compact.to_string(index=False)); print('Report generated',stats)

if __name__=='__main__': run()
