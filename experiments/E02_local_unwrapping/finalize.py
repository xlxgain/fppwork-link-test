"""Validate traceability and generate report from actual results and JUnit."""
import json,hashlib,platform,sys,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from run_e02 import ROOT,OLD,layout,plt

def run():
    out=ROOT/'outputs_E02'; review=ROOT/'outputs_C_review'
    w=pd.read_csv(out/'windows.csv'); p=pd.read_csv(out/'pairs.csv'); o=pd.read_csv(review/'pairs.csv')
    manifest=json.loads((out/'input_manifest.json').read_text())
    manifest['source_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in ROOT.glob('*.py')}
    manifest['environment']=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__)
    # Recover frozen scene centers from the historical flat placement rule once.
    params=manifest['geometry']; labels,c,inc,edges=layout(orientation='flat')
    central=int(np.argmin(np.sum((c-128)**2,axis=1)))
    i,j=next(k for k in edges if central in k and not inc[k[0]] and not inc[k[1]])
    mid=(c[i]+c[j])/2; normal=c[j]-c[i]; normal/=np.linalg.norm(normal)
    positions={'plane':None,'center':c[central],'edge':mid,'junction':mid+15*np.array([-normal[1],normal[0]])}
    for name,pixel in positions.items():
        position=None
        if pixel is not None:
            rx=(pixel[0]-params['cx'])/params['fc']; ry=(pixel[1]-params['cy'])/params['fc']
            z=params['distance_mm']/(1-params['tilt_x']*rx-params['tilt_y']*ry)
            position=[float(rx*z),float(ry*z)]
        manifest['scenes'][name].update(reference_flat_pixel=None if pixel is None else pixel.tolist(),bump_center_XY_mm=position)
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    keys=['scene_id','sigma','repeat']
    assert p.groupby(keys).input_hash.nunique().eq(1).all()
    assert o.groupby(keys).input_hash.first().equals(p.groupby(keys).input_hash.first())
    for orientation in ['flat','pointy']:
        _,_,_,edges=layout(orientation=orientation)
        assert p[p.orientation==orientation].groupby(keys+['window_method','pair_method']).size().eq(len(edges)).all()
    assert not p.loc[~p.both_windows_correct,'d_reference'].notna().any()
    w['rho']=w.rho_grid_index*2*np.pi/40
    w.to_csv(out/'windows.csv',index=False)
    for frame,path in [(p,out),(o,review)]:
        frame['predicted_variance']=np.where((frame.sigma>0)&(frame.pair_method=='M1')&frame.valid,1/frame.J,np.nan)
        frame.to_csv(path/'pairs.csv',index=False)
        frame.groupby(['orientation','scene_id','sigma','window_method','pair_method','connection','direction']).predicted_variance.mean().to_csv(path/'conditional_predicted_variance.csv')
        frame.groupby(['orientation','sigma','window_method','pair_method','reference_status']).agg(count=('valid','size'),valid_fraction=('valid','mean'),mu_mean=('mu','mean'),mu_std=('mu','std'),J_mean=('J','mean'),Q_mean=('Q','mean')).to_csv(path/'consistency_diagnostics.csv')
    example=p[(p.scene_id=='edge')&(p.sigma==5)&(p['repeat']==0)&(p.orientation=='flat')]
    fig,ax=plt.subplots(figsize=(9,4))
    for (wm,pm),g in example.groupby(['window_method','pair_method']): ax.plot(g.edge_id,g.error,'.',label=wm+'+'+pm)
    ax.set(xlabel='Fixed geometric edge ID',ylabel='Conditional offset error (cycles)'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'pair_errors.png'); plt.close(fig)
    stats=json.loads((out/'run_stats.json').read_text()); xml=ET.parse(out/'pytest.xml').getroot(); suites=list(xml.iter('testsuite'))
    tests=sum(int(s.get('tests',0)) for s in suites); failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites); seconds=sum(float(s.get('time',0)) for s in suites)
    table=w.groupby('window_method').agg(windows=('valid','size'),strict_success=('strict_success','mean'),coverage=('coverage','mean'),integer_error=('relative_integer_error_rate','mean'),continuous_rmse=('continuous_rmse','mean'))
    text=f'''# E02 无真值单窗展开与双窗接口验证

实际基线为{manifest['baseline']}，新分支experiment/E02-local-unwrapping。历史E01文件未变。固定flat参考四场景，物理位置和原几何参数记录于input_manifest.json；pointy中的center/edge/junction只是场景名，不声称仍对应其网格位置。每个(scene,sigma,repeat)只生成一次四步强度，SHA-256经核对在两朝向、所有算法及oracle对照一致；随机流独立于朝向和窗口数。噪声0/1/5，非零各20次。强度A=127.5、B=100为未饱和、未量化的浮点探索性仿真，不等同8-bit相机链路。

方法与文献差异详见methodology.md。W0确定性BFS；W1张量积Q2拟合加40点rho搜索；均不读取真值，仅访问窗内观测。T_obs只供评价及独立oracle_control。输出原样送入历史M0/M1，禁止真值对齐后输入。模2π同余只证明保留观测，不证明整数正确。

实际pytest：{tests}项，失败/错误{failures}项，用时{seconds:.3f}秒，原始日志outputs_E02/pytest.xml。完整实验耗时{stats['runtime_seconds']:.2f}秒，逐窗{stats['window_records']}条，真实单窗逐边{stats['pair_records']}条，oracle逐边{stats['oracle_records']}条。每个几何边均保留，含无效短边；覆盖率分母检查通过。最大重包裹误差{stats['max_wrap_error']:.3e}rad。

```
{table.to_string()}
```

真实单窗严格成功率{stats['strict_success_rate']:.6g}，失败窗口{stats['failures']}，两端内部正确且算法有效时双窗整数错误率{stats['integer_error_rate']:.6g}。像素加权内部错误、完整/截断窗口与边方向分组见summary_windows.csv、summary_pairs.csv和overall_summary_pairs.csv。严格成功要求单连通、完整输出、内部整数残差全零；失败像素也计入覆盖率和加权错误分母。连续RMSE在评价器只减整体整数参考后单独计算，不能解释为去噪。

当前温和物理配置下接口验证通过；若所有方法零整数错误，不支持宣称W1/M1或某朝向更优。无效截断边仍在分母。fixed_edge_error_variance先固定边跨重复计算再分组，避免模型偏差混入。两端内部不一致时唯一参考缺失，mu/J/Q仅作诊断，见consistency_diagnostics.csv；1/J不是有分支错误时完整算法的无条件方差。

C配对复核在outputs_C_review：冻结物理场景NPZ、oracle逐边完整字段与分组统计。真实单窗结果在outputs_E02：windows.csv、pairs.csv、输入清单、汇总及示例NPZ/图。未加入多窗协调或困难场景，未以制造优势为目标调参。

复现（本目录）：工作区解释器 `../../../.venv/Scripts/python.exe`；依次执行 `-m pytest -q . ../E01_hex_pairwise --junitxml=outputs_E02/pytest.xml`、`run_e02.py`、`finalize.py`。历史几何文件是冻结输入依赖。运行时间含本机实现开销，不用于算法速度排名。
'''
    (ROOT/'report_E02.md').write_text(text,encoding='utf-8'); print(table); print('Traceability checks passed; report generated')

if __name__=='__main__': run()
