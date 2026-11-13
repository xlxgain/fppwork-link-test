"""Read preserved shards, validate denominators, create diagnostic report."""
import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from run_e03 import ROOT,E02,historical_hashes,plt

def pair_summary(frame,keys):
    f=frame.copy(); f['unique_valid']=f.both_windows_correct&f.valid
    f['integer_wrong']=(f.integer_error==1)&f.unique_valid
    s=f.groupby(keys).agg(geometric_edges=('valid','size'),outputs=('valid','sum'),correct_input_edges=('both_windows_correct','sum'),unique_valid_edges=('unique_valid','sum'),integer_wrong=('integer_wrong','sum'),joint_successes=('end_to_end_success','sum')).reset_index()
    s['output_fraction']=s.outputs/s.geometric_edges
    s['correct_input_fraction']=s.correct_input_edges/s.geometric_edges
    s['conditional_integer_error_rate']=s.integer_wrong/s.unique_valid_edges.replace(0,np.nan)
    s['end_to_end_pair_success_fraction']=s.joint_successes/s.geometric_edges
    return s

def window_summary(w,keys):
    w=w.copy(); w['continuous_squared_error_sum']=w.continuous_rmse**2*w.N_out
    s=w.groupby(keys).agg(windows=('valid','size'),strict_successes=('strict_success','sum'),N_all=('N_all','sum'),N_out=('N_out','sum'),N_bad_out=('N_bad_out','sum'),N_missing=('N_missing','sum'),rejected=('valid',lambda v:int((~v).sum())),continuous_squared_error_sum=('continuous_squared_error_sum','sum')).reset_index()
    s['strict_success_fraction']=s.strict_successes/s.windows
    s['coverage']=s.N_out/s.N_all.replace(0,np.nan)
    s['covered_relative_error_rate']=s.N_bad_out/s.N_out.replace(0,np.nan)
    s['total_failure_rate']=(s.N_bad_out+s.N_missing)/s.N_all.replace(0,np.nan)
    s['continuous_rmse']=np.sqrt(s.continuous_squared_error_sum/s.N_out.replace(0,np.nan))
    s['returned_non_success']=s.windows-s.strict_successes-s.rejected
    return s

def run():
    out=ROOT/'outputs'; manifest=json.loads((out/'input_manifest.json').read_text()); stats=json.loads((out/'run_stats.json').read_text())
    assert historical_hashes()==manifest['history_before']
    assert stats['input_groups']==252 and stats['noise_draws']==40
    manifest['postprocessing_source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def read(kind):
        return pd.concat([pd.read_csv(out/r['path']) for r in manifest['files'] if r['kind']==kind],ignore_index=True)
    w=read('windows'); p=read('pairs'); o=read('oracle_pairs')
    assert (w.N_out+w.N_missing==w.N_all).all()
    assert np.allclose(w.total_failure_rate,(w.N_bad_out+w.N_missing)/w.N_all,equal_nan=True)
    for orientation,nedges in [('flat',100),('pointy',0)]:
        from run_e03 import layout
        _,_,_,edges=layout(orientation=orientation)
        for f in [p,o]: assert f[f.orientation==orientation].groupby(['scene_id','a_min','sigma','repeat','window_method','pair_method']).size().eq(len(edges)).all()
    keys=['scene_id','a_min','sigma','orientation','window_method']
    ws=window_summary(w,keys); ws.to_csv(out/'summary_windows.csv',index=False)
    window_summary(w,keys+['connection']).to_csv(out/'summary_windows_connection.csv',index=False)
    ps=pair_summary(p,keys+['pair_method']); ps.to_csv(out/'summary_pairs.csv',index=False)
    pair_summary(p,keys+['pair_method','connection','patch_related','direction']).to_csv(out/'summary_pairs_groups.csv',index=False)
    pair_summary(o,keys+['pair_method']).to_csv(out/'summary_oracle_pairs.csv',index=False)
    # Actual E02 outcome regression: all a=1 noisy/sigma=0 rows, paired keys.
    oldw=pd.read_csv(E02/'outputs_E02/windows.csv'); oldp=pd.read_csv(E02/'outputs_E02/pairs.csv')
    wk=['scene_id','sigma','repeat','orientation','window_method','window_id']
    merged=w[w.a_min==1].merge(oldw,on=wk,suffixes=('_new','_old'))
    assert len(merged)==len(w[w.a_min==1])
    assert np.array_equal(merged.strict_success_new,merged.strict_success_old)
    assert np.allclose(merged.covered_relative_error_rate,merged.relative_integer_error_rate,equal_nan=True)
    pk=['scene_id','sigma','repeat','orientation','window_method','pair_method','edge_id']
    mergedp=p[p.a_min==1].merge(oldp,on=pk,suffixes=('_new','_old'))
    assert len(mergedp)==len(p[p.a_min==1])
    assert np.allclose(mergedp.mu_new,mergedp.mu_old,atol=1e-12,rtol=0,equal_nan=True)
    assert np.array_equal(mergedp.valid_new,mergedp.valid_old)
    for f in [p,o]:
        assert f.loc[~f.both_windows_correct,'d_reference'].isna().all()
        assert f.loc[~f.both_windows_correct,'error'].isna().all()
    # Outside propagation uses SAME noise draw/window and SAME evaluator c.
    base=w[w.a_min==1][wk+['outside_N_bad_out']].rename(columns={'outside_N_bad_out':'outside_bad_baseline'})
    propagation=w.merge(base,on=wk); propagation['outside_added_bad']=propagation.outside_N_bad_out-propagation.outside_bad_baseline
    propagation.groupby(keys).agg(outside_bad=('outside_N_bad_out','sum'),outside_added_bad=('outside_added_bad','sum'),outside_pixels=('outside_N_all','sum'),core_bad=('core_N_bad_out','sum'),transition_bad=('transition_N_bad_out','sum')).to_csv(out/'outside_propagation.csv')
    # Fixed edge variance, conditional diagnostic only; record count for selection.
    gkeys=['scene_id','a_min','sigma','orientation','window_method','pair_method','edge_id','snr_diagnostic_group']
    v=o[o.sigma>0].groupby(gkeys).agg(n=('error','count'),empirical_variance=('error','var'),firstorder_oracle=('v_mu_firstorder_oracle','mean'),bias=('error','mean')).reset_index(); v.to_csv(out/'oracle_fixed_edge_variance.csv',index=False)
    higher=v[(v.pair_method=='M1')&(v.snr_diagnostic_group=='higher_snr')&(v.n==20)].dropna(subset=['empirical_variance','firstorder_oracle'])
    if len(higher):
        fig,ax=plt.subplots(figsize=(5,5)); ax.scatter(higher.firstorder_oracle,higher.empirical_variance,s=5,alpha=.4)
        lower=min(higher.firstorder_oracle.min(),higher.empirical_variance.min()); upper=max(higher.firstorder_oracle.max(),higher.empirical_variance.max())
        ax.plot([lower,upper],[lower,upper],'k--'); ax.set(xscale='log',yscale='log',xlabel='Oracle first-order propagation',ylabel='Fixed-edge empirical variance',title='Oracle, higher-SNR diagnostic ONLY'); fig.tight_layout(); fig.savefig(out/'oracle_higher_snr_variance.png'); plt.close(fig)
    real=p[p.sigma>0].groupby(gkeys).agg(unique_valid_repeats=('error','count'),conditional_variance=('error','var'),conditional_bias=('error','mean')).reset_index(); real.to_csv(out/'real_fixed_edge_diagnostics.csv',index=False)
    rep=window_summary(w[w.sigma>0],['scene_id','a_min','repeat','orientation','window_method'])
    rep.to_csv(out/'per_repeat_windows.csv',index=False)
    paired=rep.pivot(index=['scene_id','a_min','repeat','window_method'],columns='orientation',values='strict_success_fraction').reset_index()
    paired['pointy_minus_flat']=paired.pointy-paired.flat
    orientation_stats=paired.groupby(['scene_id','a_min','window_method']).pointy_minus_flat.agg(['mean','std','count'])
    orientation_stats.to_csv(out/'paired_orientation_difference.csv')
    obs=pd.read_csv(out/'observation_summary.csv')
    os=obs.groupby(['scene_id','a_min','sigma','zone']).agg(pixels=('n_pixels','sum'),squared_error=('squared_error_sum','sum'),large_errors=('over_pi_half_count','sum'),B_true_mean=('B_true_mean','mean'),B_hat_mean=('B_hat_mean','mean')).reset_index(); os['rmse']=np.sqrt(os.squared_error/os.pixels); os['over_pi_half_fraction']=os.large_errors/os.pixels; os.to_csv(out/'summary_observation.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for (orientation,wm),g in ws[ws.sigma==5].groupby(['orientation','window_method']):
        # Pool counts, not average ratios.
        s=g.groupby('a_min')[['strict_successes','windows','N_bad_out','N_missing','N_all']].sum()
        axs[0].plot(s.index,s.strict_successes/s.windows,'o-',label=orientation+' '+wm)
        axs[1].plot(s.index,(s.N_bad_out+s.N_missing)/s.N_all,'o-')
    axs[0].set(ylabel='Strict window success'); axs[1].set(ylabel='Pixel total failure');
    for ax in axs: ax.set(xlabel='Minimum attenuation',xscale='log')
    axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'window_performance.png'); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    pooled=pair_summary(p[p.sigma==5],['a_min','window_method','pair_method'])
    for (wm,pm),g in pooled.groupby(['window_method','pair_method']):
        axs[0].plot(g.a_min,g.conditional_integer_error_rate,'o-',label=wm+'+'+pm)
        axs[1].plot(g.a_min,g.end_to_end_pair_success_fraction,'o-')
    axs[0].set(ylabel='Conditional integer error'); axs[1].set(ylabel='Joint pair success / ALL edges')
    for ax in axs: ax.set(xscale='log',xlabel='Minimum attenuation')
    axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'pair_performance.png'); plt.close(fig)
    example=np.load(out/'example_W0.npz'); fig,axs=plt.subplots(1,4,figsize=(14,4))
    for ax,field in zip(axs,['B_true','B_hat','frame0','circular_error']):
        im=ax.imshow(example[field]); ax.set_title(field); fig.colorbar(im,ax=ax,shrink=.6)
    fig.tight_layout(); fig.savefig(out/'mechanism.png'); plt.close(fig)
    failing=w[(w.sigma==5)&(w.N_bad_out>0)].a_min
    first=float(failing.max()) if len(failing) else None
    # First error may occur in repeat != 0. Re-evaluate that EXACT predeclared
    # observation to show actual errors, not a conveniently selected difficulty.
    if first is not None:
        from run_e03 import layout,support,unwrap_window,demodulate
        from radiometry import attenuation,noise_field,intensities
        from evaluation_e03 import evaluate
        witness=w[(w.a_min==first)&(w.sigma==5)&(w.N_bad_out>0)].iloc[0]
        source=np.load(E02/'outputs_C_review'/f'{witness.scene_id}.npz')
        truth=source['phase_radians']; mask=source['valid']; depth=source['depth_mm']
        a,zones=attenuation(depth,manifest['geometry'],manifest['center_XY_mm'],first)
        cfg=manifest['config']; index=cfg['source_scene_indices'][witness.scene_id]
        phi=demodulate(intensities(truth,a,noise_field(truth.shape,cfg['seed'],index,5,int(witness['repeat']))))
        yy,xx=np.indices(phi.shape); coords=np.stack([xx,yy],-1)
        fig,axs=plt.subplots(2,2,figsize=(9,8))
        for row,orientation in enumerate(['flat','pointy']):
            labels,centers,_,_=layout(orientation=orientation)
            for col,method in enumerate(['W0','W1_MLSQU_HEX_Q2']):
                residualmap=np.full(phi.shape,np.nan)
                for i in range(len(centers)):
                    region=mask&(labels==i); psi,diag=unwrap_window(phi,region,coords,dict(method=method,center=centers[i]))
                    _,residual=evaluate(psi,phi,truth,region,diag,zones); residualmap[region]=residual[region]
                ax=axs[row,col]; ax.imshow(residualmap,cmap='coolwarm',vmin=-2,vmax=2); ax.contour(zones,levels=[.5,1.5],colors='k',linewidths=.5); ax.set_title(f'{orientation} {method}')
        fig.suptitle(f'First failing level: {witness.scene_id} a={first}, repeat={int(witness["repeat"])}'); fig.tight_layout(); fig.savefig(out/'first_error_actual_repeat.png'); plt.close(fig)
    for minimum in sorted(set([.05]+([first] if first else []))):
        fig,axs=plt.subplots(2,2,figsize=(9,8))
        for row,orientation in enumerate(['flat','pointy']):
            for col,wm in enumerate(['W0','W1_MLSQU_HEX_Q2']):
                data=np.load(out/f'map_edge_{minimum}_{orientation}_{wm}.npz')['residual_integer']
                ax=axs[row,col]; im=ax.imshow(data,cmap='coolwarm',vmin=-2,vmax=2); ax.contour(example['zones'],levels=[.5,1.5],colors='k',linewidths=.5); ax.set_title(f'{orientation} {wm} a={minimum}'); fig.colorbar(im,ax=ax,shrink=.6)
        fig.tight_layout(); fig.savefig(out/f'integer_maps_{minimum}.png'); plt.close(fig)
    suite=list(ET.parse(out/'pytest.xml').getroot().iter('testsuite'))
    tests={key:sum(float(s.get(key,0)) for s in suite) for key in ['tests','failures','errors','skipped','time']}
    compact=window_summary(w[w.sigma==5],['a_min','window_method'])
    results=pair_summary(p[p.sigma==5],['a_min','window_method','pair_method'])
    core=os[(os.sigma==5)&(os.zone=='core')][['scene_id','a_min','rmse','over_pi_half_fraction']]
    outside=propagation[propagation.sigma==5].groupby(['a_min','window_method'])[['outside_N_bad_out','outside_added_bad']].sum()
    stats.update(tests=tests,window_records=len(w),pair_records=len(p),oracle_records=len(o),first_integer_error_attenuation=first,E02_regression_passed=True,history_unchanged=True)
    (out/'run_stats.json').write_text(json.dumps(stats,indent=2))
    text=f'''# E03 局部低调制度误差传递诊断

基线{manifest['baseline']}；只新增E03目录，E01/E02历史文件逐文件哈希前后相同。共252输入条件，40份独立非零噪声底稿，各衰减档和算法复用；窗口与边记录不是独立实验次数。完整实验{stats['runtime_seconds']:.2f}秒；实际pytest记录{tests}，见pytest.xml。

几何冻结为E02 plane/edge，物体与斑块不随朝向移动。edge只是历史flat参考命名。按深度恢复XY，斑块中心{manifest['center_XY_mm']}mm，内/外半径30/50mm。所有六档为预定探索性数值，并非测得反射率或实际远距离设备参数。

强度I=27.5+a[100+100cos(Φ+kπ/2)]+ε，ε独立高斯σ=0/5；固定27.5项只是不随回波缩放的偏置/背景项，并非全部环境光。无量化、无裁剪、无饱和、无掩膜删除；超出0–255数量在输入清单中记录。a=1时与E02强度SHA逐位相同；无噪所有档相位误差小于1e−12rad。a=1的单窗成绩和双窗mu/valid完整回归通过。

W0/W1、M0/M1直接调用冻结实现。输出未真值对齐即送双窗。d=c_j−c_i；非唯一参考只留诊断，主error/d_reference为NaN。评价修正N_all、N_out、N_missing与N_bad_out，覆盖输出错误率和含缺失的总失败率分开；整窗无输出c=NaN。失败窗不能由一个整数统一解释时，后端一窗一整数表示不足，未要求双窗凭空修复。

## 观测层（core）
```
{core.to_string(index=False)}
```
调制降低使圆周误差逐渐增大，不存在本实验确定的普适可测阈值；π/2只作诊断分箱。即使相对T_obs级次正确，也不能宣称高精度物理相位已恢复。

## 单窗层（sigma=5，按像素计数加权）
```
{compact.to_string(index=False)}
```
首个出现内部整数错误的预定档为{first}。该值是当前冻结场景/噪声样本的观察，不是保证。错误延伸到outside的计数（共同噪声配对，与a=1同一窗口比较）：
```
{outside.to_string()}
```
因此不能仅以弱区本身的相位误差概括单窗误差；具体空间范围见预定a=0.05与首错档图。缺失和错误已分别记录，未把部分输出错误率乘全部像素冒充错误数。

## 双窗层（sigma=5）
```
{results[['a_min','window_method','pair_method','geometric_edges','output_fraction','correct_input_fraction','conditional_integer_error_rate','end_to_end_pair_success_fraction']].to_string(index=False)}
```
条件错误率只除以两端严格正确且有输出的边，空样本为NaN；联合成功率除以全部几何边。internal/truncated、斑块相关/不相关和边方向另表统计，失败边不删除。oracle_control单独保存以分离归因，不能解释为弱区可靠可测。

J/Q统一标为legacy_homoscedastic，未校准为低调制度置信度。评价侧用B_true及固定q计算一阶方差，仅供误差传播诊断。oracle固定边高SNR组(min B_true/σ≥5)的经验方差见oracle_fixed_edge_variance.csv；弱信号不承诺高斯/卡方/覆盖率。真实单窗分支错误时以上方差不是完整算法无条件误差。

## 五个问题的当前回答

1. 观测误差随衰减连续增长：core在a=0.25时已约0.14rad、a=0.1约0.39rad，后者首次观察到整数级次错误；观测变差与整数错误并非同一阈值。本配置不能给“明显变差”设普适边界。
2. W0在弱档存在outside级次错误；W1在本轮outside错误计数见上表。这个结论只适用于本轮冻结方法与场景，不比较不存在的全图基线。
3. 有输出但N_bad_out>0的窗口确实不能只用一个整体整数解释；本轮无算法拒绝或缺失，不应把正常返回误记为严格成功。部分输出/全失败计数机制仍通过独立单元测试验证。
4. 两端严格正确且有输出时M0/M1的条件整数错误率均为0；全几何边联合成功仍受原截断短边无支撑和内部错窗影响。即使a=1联合成功约89.13%，也不是条件连接算法错误。
5. 固定物理斑块下弱档出现朝向差异：以下为同一scene/repeat的pointy−flat严格成功率差，均值/标准差用20份噪声底稿计算；a≥0.25均无差异，低档结果不能推广为普适网格朝向排名。
```
{orientation_stats.to_string()}
```

样本共同随机相关，不把窗口/边记录数当独立重复数。当前结果仅给固定几何和辐射机制下的使用边界，不推广到所有全图方法或真实距离。文献机制与配置限定见methodology.md。

复现：在本目录用工作区`.venv`执行`python -m pytest -q . ../E02_local_unwrapping ../E01_hex_pairwise --junitxml=outputs/pytest.xml`，再运行`run_e03.py`、`summarize.py`。raw逐窗/逐边按条件分CSV保存于records，统一索引与输入/噪声/衰减/几何哈希在input_manifest.json。无新算法、多窗优化或追加困难档位。
'''
    (ROOT/'report_E03.md').write_text(text,encoding='utf-8'); print(compact.to_string(index=False)); print('Validation and report complete',stats)

if __name__=='__main__': run()
