import json,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from run_e04 import ROOT

def main():
    out=ROOT/'outputs'; w=pd.read_csv(out/'windows.csv'); p=pd.read_csv(out/'pairs.csv'); stats=json.loads((out/'run_stats.json').read_text()); cfg=json.loads((ROOT/'config.json').read_text())
    assert w.true_integer_error_rate.isna().all() and p.true_integer_error_rate.isna().all() and p.d_reference.isna().all()
    assert ((w.N_all-w.N_out)==w.N_missing).all()
    grouped=w.groupby(['orientation','window_method']).agg(windows=('valid','size'),accepted=('valid','sum'),N_geometric=('N_geometric','sum'),N_valid=('N_all','sum'),N_out=('N_out','sum'),N_missing=('N_missing','sum'),mean_neighbor_jump_fraction=('neighbor_integer_jump_fraction','mean')).reset_index()
    grouped['coverage_on_valid']=grouped.N_out/grouped.N_valid
    grouped['coverage_on_all_pixels']=grouped.N_out/grouped.N_geometric
    grouped.to_csv(out/'overall_windows.csv',index=False)
    gs=p.groupby(['orientation','window_method','pair_method']).agg(geometric_edges=('valid','size'),valid_edges=('valid','sum'),mu_mean=('mu','mean'),mu_std=('mu','std')).reset_index(); gs['valid_fraction']=gs.valid_edges/gs.geometric_edges; gs.to_csv(out/'overall_pairs.csv',index=False)
    assert grouped.N_geometric.eq(stats['total_pixels']).all()
    for ori,part in p.groupby('orientation'):
        assert part.groupby(['window_method','pair_method']).edge_id.nunique().nunique()==1
    suites=list(ET.parse(out/'pytest.xml').getroot().iter('testsuite')); tests={key:sum(float(s.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped','time']}
    checks=pd.read_csv(out/'capture_checks.csv'); refs=pd.read_csv(out/'reference_checks.csv'); zones=pd.read_csv(out/'observation_zones.csv')
    stats['pytest']=tests; (out/'run_stats.json').write_text(json.dumps(stats,indent=2))
    text=f'''# E04 真实单目条纹数据过渡验证

## 数据核查

来源：[hectorcs23/fringe-projection-3d-reconstruction]({cfg['source']})，版本`{cfg['source_commit']}`，MIT许可及归属见SOURCE_LICENSE.txt。作者README说明Canon M50原始拍摄4608×3072；公开照片已经缩小到1152×768且为JPEG压缩，**实拍但不是原始传感器数据**。本实验对所有32张sphere图像只做Pillow灰度转换，没有resize、重采样或额外下采样；完整1152×768参与计算。图中局部放大只是显示裁窗。

作者src/sphere_multifreq.py指定参考平面DSC_0169–0184，球面DSC_0185–0200；各按连续四张分成四频，顺序0、π/2、π、3π/2。该顺序由作者命名及代码支持，不声称经过独立硬件相移标定。球面脚本计数[2,4,14,60]；README [2,4,18,70]对应项目概述，不混用。逐文件组别、步进、尺寸、SHA-256和源码版本在source_manifest.json。FFT核查如下（中央条带半幅频谱估计只验证粗频率，不等同精确投影频率标定）：
```
{checks.to_string(index=False)}
```
四步余弦拟合存在非零残差，不能将实拍链路视为理想四步相移；正反帧和差残差、拟合RMSE均已保存。JPEG、曝光、相移误差与物体运动等可能因素未分离识别，不作无依据归因。

## 观测与有效掩膜

phi=atan2(I3−I1,I0−I2)，A=mean(Ik)，B=0.5hypot(I0−I2,I3−I1)。采用作者已有风格的B>0.1×P99(B)、15<A<250，仅球面最高频观测决定W0/W1有效域；参考平面只用于多频诊断。有效像素{stats['valid_pixels']}/{stats['total_pixels']}（{stats['valid_pixels']/stats['total_pixels']:.2%}）。掩膜可能断开窗口，按历史算法显式拒绝，不填洞。低调制度像素单独显示并计数，不把掩膜删除像素伪装算法成功。
```
{zones.to_string(index=False)}
```
低调制度定义为曝光范围内且B低于同一阈值；曝光范围外另在总分母体现。球面显示圆心/半径({cfg['sphere_display_center_xy']},{cfg['sphere_display_radius_px']})是人工读图的展示区域，边界带±8px，不是几何真值、不参与算法或选点。edge_low_modulation.png与real_observations.png展示实拍、相位和幅值。

## 多频独立参考尝试与拒绝准确率声明

沿作者物体减参考平面定义Δphi_k=wrap(phi_obj,k−phi_ref,k)，按[2,4,14,60]层次预测并选择相位分支，**不使用W0/W1输出**。同像素相减沿用作者的固定机位对应假设，未独立标定跨组配准。为了与单窗的物体绝对相位定义比较，另用参考平面逐轴numpy.unwrap的carrier加回ΔΦ，仅构成候选；最低频Δphi是否无包裹、频率精确比例/相位原点、平面carrier支路都没有独立标定保证。因此本轮可信整数参考像素数为0。
```
{refs.to_string(index=False)}
```
候选对最高频的模2π最大残差为{stats['max_candidate_modulo_error']:.3e}rad，但这是候选构造的代数结果，不验证级次。最后一阶跨频预测RMSE约0.227rad，且部分位置更差；不能以自洽或阈值筛选替代外部认证。所有true_integer_error_rate与d_reference均为NaN，不公布“真实整数准确率”。windows.csv的candidate_integer_inconsistency仅表示与未验证候选参考的窗内模态级次不一致，不能归因于W0/W1错误。邻接普通差相对包裹梯度的整数跳变也是诊断，物体边缘真实不连续同样可能产生它。

## 冻结单窗与双窗接入

直接调用E02的W0/W1_MLSQU_HEX_Q2、E01的estimate(M0/M1)和support，边长30px、法向半宽6px、端点留白6px、W1张量积8项/40rho保持不变。全图规则六边形扩展了中心索引范围以覆盖1152×768；窗口/支撑仅在所需包围盒计算，同一实际坐标和像素不改变。测试核对256基准分区与原几何一致、局部裁窗与全图算法输出等价。无全图展开后再裁窗，无补洞或质量引导回退。
```
{grouped.to_string(index=False)}
```
覆盖率同时以有效像素和全部原图像素作分母；空窗、断连和秩亏窗口保留，失败不删除。重包裹一致仅验证观测同余，不证明只剩一个整体整数偏移；在无可信参考情况下无法给这一性质作真实准确率认证。

双窗结果（全部几何边为分母）：
```
{gs.to_string(index=False)}
```
完整/截断、球面展示边界相关/不相关、边方向等逐边字段保存在pairs.csv及summary_pairs.csv；每条几何边每种组合均保留。mu是窗口参考约定下的整数偏移测量，不是物理深度。J/Q以单位相位方差输出，仅为几何/残差诊断，不是经过实拍噪声校准的置信度。无法确认唯一d_reference时不计算偏移真实误差。低调制区、球面边缘和候选参考不一致的位置不据此归为算法真错。

## 实际执行与结论

实际pytest：{tests}；原始记录outputs/pytest.xml。完整实验运行{stats['runtime_seconds']:.2f}秒，逐窗{stats['window_records']}条、逐边{stats['pair_records']}条。32张JPEG均实际加载，未复制外部数据集入本仓库；data_cache被gitignore。处理代码、来源/许可、CSV统计、少量图与一条轻量相位剖面NPZ提交。原E01–E03文件未改。

本轮完成真实观测到冻结单窗/双窗的接口与诊断验证。没有可信独立整数真值，不能据此宣称某方法更准确、闭合意味着真实正确或获得真实三维测量精度；未实现双目几何、新算法或大型参数搜索。

复现：工作区.venv解释器运行`fetch_data.py`获取固定外部提交，再执行`python -m pytest -q . ../E03_low_modulation ../E02_local_unwrapping ../E01_hex_pairwise --junitxml=outputs/pytest.xml`、`run_e04.py`、`report.py`。
'''
    (ROOT/'report_E04.md').write_text(text,encoding='utf-8'); print(grouped.to_string(index=False)); print(gs.to_string(index=False)); print('Report and denominator checks complete')

if __name__=='__main__': main()
