# E04 真实单目条纹数据过渡验证

## 数据核查

来源：[hectorcs23/fringe-projection-3d-reconstruction](https://github.com/hectorcs23/fringe-projection-3d-reconstruction)，版本`35967405441b58addd2a2c9e877ee4c3d4821ccc`，MIT许可及归属见SOURCE_LICENSE.txt。作者README说明Canon M50原始拍摄4608×3072；公开照片已经缩小到1152×768且为JPEG压缩，**实拍但不是原始传感器数据**。本实验对所有32张sphere图像只做Pillow灰度转换，没有resize、重采样或额外下采样；完整1152×768参与计算。图中局部放大只是显示裁窗。

作者src/sphere_multifreq.py指定参考平面DSC_0169–0184，球面DSC_0185–0200；各按连续四张分成四频，顺序0、π/2、π、3π/2。该顺序由作者命名及代码支持，不声称经过独立硬件相移标定。球面脚本计数[2,4,14,60]；README [2,4,18,70]对应项目概述，不混用。逐文件组别、步进、尺寸、SHA-256和源码版本在source_manifest.json。FFT核查如下（中央条带半幅频谱估计只验证粗频率，不等同精确投影频率标定）：
```
     kind  frequency_index  fit_rmse  opposite_sum_rmse  fft_central_strip_fullwidth
reference                0 10.860223          21.720447                          2.0
reference                1 10.365236          20.730473                          4.0
reference                2  8.269512          16.539024                         14.0
reference                3  5.266800          10.533601                         60.0
   object                0 10.785055          21.570110                          2.0
   object                1 10.179345          20.358691                          4.0
   object                2  8.084501          16.169001                         14.0
   object                3  5.113496          10.226993                         60.0
```
四步余弦拟合存在非零残差，不能将实拍链路视为理想四步相移；正反帧和差残差、拟合RMSE均已保存。JPEG、曝光、相移误差与物体运动等可能因素未分离识别，不作无依据归因。

## 观测与有效掩膜

phi=atan2(I3−I1,I0−I2)，A=mean(Ik)，B=0.5hypot(I0−I2,I3−I1)。采用作者已有风格的B>0.1×P99(B)、15<A<250，仅球面最高频观测决定W0/W1有效域；参考平面只用于多频诊断。有效像素763054/884736（86.25%）。掩膜可能断开窗口，按历史算法显式拒绝，不填洞。低调制度像素单独显示并计数，不把掩膜删除像素伪装算法成功。
```
                   zone  pixels  valid_pixels     A_mean    B_mean     B_p10  closure_mean
                    all  884736        763054  71.212309 37.489615  0.500000  6.548987e-14
                  valid  763054        763054  81.990855 43.388983 29.769951  7.101213e-14
         low_modulation     808             0  24.024443  4.199896  2.915476  5.508473e-14
sphere_boundary_display    8772          8689 100.208048 35.840069 11.401754  1.318257e-13
sphere_interior_display   19573         19573  94.421678 38.894022 24.351591  1.352464e-13
```
低调制度定义为曝光范围内且B低于同一阈值；曝光范围外另在总分母体现。球面显示圆心/半径([572, 414],87)是人工读图的展示区域，边界带±8px，不是几何真值、不参与算法或选点。edge_low_modulation.png与real_observations.png展示实拍、相位和幅值。

## 多频独立参考尝试与拒绝准确率声明

沿作者物体减参考平面定义Δphi_k=wrap(phi_obj,k−phi_ref,k)，按[2,4,14,60]层次预测并选择相位分支，**不使用W0/W1输出**。同像素相减沿用作者的固定机位对应假设，未独立标定跨组配准。为了与单窗的物体绝对相位定义比较，另用参考平面逐轴numpy.unwrap的carrier加回ΔΦ，仅构成候选；最低频Δphi是否无包裹、频率精确比例/相位原点、平面carrier支路都没有独立标定保证。因此本轮可信整数参考像素数为0。
```
 stage    ratio      n  prediction_circular_rmse  fraction_abs_residual_under_02  trusted_pixels                                       status
     1 2.000000 762150                  0.064515                        0.988916               0 unverified_frequency_origin_and_lowest_order
     2 3.500000 762037                  0.131546                        0.957473               0 unverified_frequency_origin_and_lowest_order
     3 4.285714 760161                  0.227043                        0.713346               0 unverified_frequency_origin_and_lowest_order
```
候选对最高频的模2π最大残差为4.802e-13rad，但这是候选构造的代数结果，不验证级次。最后一阶跨频预测RMSE约0.227rad，且部分位置更差；不能以自洽或阈值筛选替代外部认证。所有true_integer_error_rate与d_reference均为NaN，不公布“真实整数准确率”。windows.csv的candidate_integer_inconsistency仅表示与未验证候选参考的窗内模态级次不一致，不能归因于W0/W1错误。邻接普通差相对包裹梯度的整数跳变也是诊断，物体边缘真实不连续同样可能产生它。

## 冻结单窗与双窗接入

直接调用E02的W0/W1_MLSQU_HEX_Q2、E01的estimate(M0/M1)和support，边长30px、法向半宽6px、端点留白6px、W1张量积8项/40rho保持不变。全图规则六边形扩展了中心索引范围以覆盖1152×768；窗口/支撑仅在所需包围盒计算，同一实际坐标和像素不改变。测试核对256基准分区与原几何一致、局部裁窗与全图算法输出等价。无全图展开后再裁窗，无补洞或质量引导回退。
```
orientation   window_method  windows  accepted  N_geometric  N_valid  N_out  N_missing  mean_neighbor_jump_fraction  coverage_on_valid  coverage_on_all_pixels
       flat              W0      419       369       884736   763054 761086       1968                          0.0           0.997421                0.860241
       flat W1_MLSQU_HEX_Q2      419       367       884736   763054 760991       2063                          0.0           0.997296                0.860133
     pointy              W0      443       361       884736   763054 761447       1607                          0.0           0.997894                0.860649
     pointy W1_MLSQU_HEX_Q2      443       360       884736   763054 761394       1660                          0.0           0.997825                0.860589
```
覆盖率同时以有效像素和全部原图像素作分母；空窗、断连和秩亏窗口保留，失败不删除。重包裹一致仅验证观测同余，不证明只剩一个整体整数偏移；在无可信参考情况下无法给这一性质作真实准确率认证。

双窗结果（全部几何边为分母）：
```
orientation   window_method pair_method  geometric_edges  valid_edges   mu_mean   mu_std  valid_fraction
       flat              W0          M0             1172          997 -1.562688 1.165662        0.850683
       flat              W0          M1             1172          996 -1.564229 1.165141        0.849829
       flat W1_MLSQU_HEX_Q2          M0             1172          995 -1.529648 1.160151        0.848976
       flat W1_MLSQU_HEX_Q2          M1             1172          994 -1.530157 1.160542        0.848123
     pointy              W0          M0             1206          986 -0.937120 1.745219        0.817579
     pointy              W0          M1             1206          984 -0.933921 1.745488        0.815920
     pointy W1_MLSQU_HEX_Q2          M0             1206          985 -0.924873 1.731006        0.816750
     pointy W1_MLSQU_HEX_Q2          M1             1206          983 -0.921648 1.731227        0.815091
```
完整/截断、球面展示边界相关/不相关、边方向等逐边字段保存在pairs.csv及summary_pairs.csv；每条几何边每种组合均保留。mu是窗口参考约定下的整数偏移测量，不是物理深度。J/Q以单位相位方差输出，仅为几何/残差诊断，不是经过实拍噪声校准的置信度。无法确认唯一d_reference时不计算偏移真实误差。低调制区、球面边缘和候选参考不一致的位置不据此归为算法真错。

## 实际执行与结论

实际pytest：{'tests': 37.0, 'failures': 0.0, 'errors': 0.0, 'skipped': 0.0, 'time': 5.84}；原始记录outputs/pytest.xml。完整实验运行23.44秒，逐窗1724条、逐边9512条。32张JPEG均实际加载，未复制外部数据集入本仓库；data_cache被gitignore。处理代码、来源/许可、CSV统计、少量图与一条轻量相位剖面NPZ提交。原E01–E03文件未改。

本轮完成真实观测到冻结单窗/双窗的接口与诊断验证。没有可信独立整数真值，不能据此宣称某方法更准确、闭合意味着真实正确或获得真实三维测量精度；未实现双目几何、新算法或大型参数搜索。

复现：工作区.venv解释器运行`fetch_data.py`获取固定外部提交，再执行`python -m pytest -q . ../E03_low_modulation ../E02_local_unwrapping ../E01_hex_pairwise --junitxml=outputs/pytest.xml`、`run_e04.py`、`report.py`。
