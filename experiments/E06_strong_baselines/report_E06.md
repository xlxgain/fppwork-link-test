# E06 强空间基线与可靠初始化多频对照

基线1131f910603800b501996844fbd5ba38a72e34cb；E01–E05逐文件SHA前后相同。全部252组预定输入，3528条整图记录；每个方法每次保留失败，不删除弱档。完整运行844.04秒。实际pytest记录{'tests': 48.0, 'failures': 0.0, 'errors': 0.0, 'skipped': 0.0, 'time': 14.144}，日志pytest.xml。非零噪声底稿每频40份、七频共280份，最高频只有40份；六档/算法成绩配对相关，不能把像素或3528记录当独立重复数。

## 强基线、频率与公平性

H2002为scikit-image 0.25.2的二维unwrap_phase，实际签名(image, wrap_around=False, rng=None)，rng=0、wrap_around=False，masked array屏蔽无效；逐共同连通分量保留零点。输出先核对同余，再表达成原phi加整数周期，不以投影隐藏错误实现。具体论文与库依据见methodology.md。

8/12/16/20帧依次使用[1,60]、[1,8,60]、[1,4,15,60]、[1,3,8,22,60]。最低频在已知投影域和零原点下无噪初始化通过；弱信号时不保证分支正确。所有频率均实际生成四步；算法只读取自身频率，七频池28帧不是每算法帧数。无E05频率2起始歧义基线进入本轮排名。

平面和(0,0)mm光滑凸包保持E05针孔几何，最高f=60。采用E03精确物理斑块中心[-43.69148144552595, -24.805602372239356]mm、30/50mm半径、六档a。I=(27.5+100a)+100a cos(Φf+kπ/2)+ε，σ=0/5，非零各20次。浮点不裁剪不量化；不引入各频额外MTF，不能推广为真实系统各频同成像。固定背景不是全部环境反射光。参数为预定探索性机制配置。

所有方法共用原几何mask，低调制不提前删除。a=1最高频哈希和E05结果回归通过（仅共同输入分量一个整数参考，不逐窗对齐）。八种六边形组合与E05整数图协调直接调用、原选点和参数不变；主方法预先指定flat-W1-M1。原M1 J/Q低调制下未校准，图仍round(mu)等权L1，不使用新增置信权。

## 评价与原始表

e=round((Psi−T_obs)/2π)，只对输入mask四连通分量各减一个众数；core/transition/outside/patch均沿用此误差图，不独立对齐。总错误=(已输出错+缺失)/全部几何像素；covered-only、覆盖率、相位RMSE独立给出。四帧无锚绝对能力标“整体整数自由度未消除”；原始零点差仅诊断，不用0%给四帧算法作失败排名。外部中心(128,128)锚点附表明确是评分端真值模拟校准，不是观测自身能力。

主方法全图表：
```
 scene  a_min  error_total  phase_rmse  coverage  whole_success
 plane   0.02 4.747772e-03    0.412632       1.0              0
 plane   0.05 9.307861e-05    0.107324       1.0              0
 plane   0.10 0.000000e+00    0.057444       1.0             20
 plane   0.25 0.000000e+00    0.039316       1.0             20
 plane   0.50 0.000000e+00    0.036293       1.0             20
 plane   1.00 0.000000e+00    0.035383       1.0             20
smooth   0.02 4.825592e-03    0.416318       1.0              0
smooth   0.05 1.129150e-04    0.108561       1.0              0
smooth   0.10 7.629395e-07    0.057274       1.0             19
smooth   0.25 0.000000e+00    0.039322       1.0             20
smooth   0.50 0.000000e+00    0.036285       1.0             20
smooth   1.00 0.000000e+00    0.035370       1.0             20
```
核心patch对照表（全部档位，不删失败）：
```
 scene  a_min                  method  error_total  coverage  phase_rmse
 plane   0.02           H2002_skimage     0.136771       1.0    2.943821
 plane   0.02      TPU2_unit_balanced     0.645391       1.0   59.546663
 plane   0.02      TPU3_unit_balanced     0.428700       1.0   60.262201
 plane   0.02      TPU4_unit_balanced     0.392327       1.0   61.635907
 plane   0.02      TPU5_unit_balanced     0.379646       1.0   65.038491
 plane   0.02 flat_W1_MLSQU_HEX_Q2_M1     0.154484       1.0    2.344905
 plane   0.05           H2002_skimage     0.016517       1.0    0.960865
 plane   0.05      TPU2_unit_balanced     0.627005       1.0   35.792515
 plane   0.05      TPU3_unit_balanced     0.359716       1.0   36.916067
 plane   0.05      TPU4_unit_balanced     0.266492       1.0   38.429808
 plane   0.05      TPU5_unit_balanced     0.216442       1.0   39.926306
 plane   0.05 flat_W1_MLSQU_HEX_Q2_M1     0.003039       1.0    0.580070
 plane   0.10           H2002_skimage     0.000125       1.0    0.266536
 plane   0.10      TPU2_unit_balanced     0.592750       1.0   15.708171
 plane   0.10      TPU3_unit_balanced     0.199302       1.0   17.063225
 plane   0.10      TPU4_unit_balanced     0.052491       1.0   14.213211
 plane   0.10      TPU5_unit_balanced     0.019133       1.0   11.295093
 plane   0.10 flat_W1_MLSQU_HEX_Q2_M1     0.000000       1.0    0.260978
 plane   0.25           H2002_skimage     0.000000       1.0    0.104072
 plane   0.25      TPU2_unit_balanced     0.488540       1.0    6.460725
 plane   0.25      TPU3_unit_balanced     0.004883       1.0    2.641247
 plane   0.25      TPU4_unit_balanced     0.000000       1.0    0.104072
 plane   0.25      TPU5_unit_balanced     0.000000       1.0    0.104072
 plane   0.25 flat_W1_MLSQU_HEX_Q2_M1     0.000000       1.0    0.104072
 plane   0.50           H2002_skimage     0.000000       1.0    0.058033
 plane   0.50      TPU2_unit_balanced     0.333782       1.0    3.845584
 plane   0.50      TPU3_unit_balanced     0.000000       1.0    0.058033
 plane   0.50      TPU4_unit_balanced     0.000000       1.0    0.058033
 plane   0.50      TPU5_unit_balanced     0.000000       1.0    0.058033
 plane   0.50 flat_W1_MLSQU_HEX_Q2_M1     0.000000       1.0    0.058033
 plane   1.00           H2002_skimage     0.000000       1.0    0.035196
 plane   1.00      TPU2_unit_balanced     0.139387       1.0    2.345441
 plane   1.00      TPU3_unit_balanced     0.000000       1.0    0.035196
 plane   1.00      TPU4_unit_balanced     0.000000       1.0    0.035196
 plane   1.00      TPU5_unit_balanced     0.000000       1.0    0.035196
 plane   1.00 flat_W1_MLSQU_HEX_Q2_M1     0.000000       1.0    0.035196
smooth   0.02           H2002_skimage     0.171315       1.0    3.756164
smooth   0.02      TPU2_unit_balanced     0.640886       1.0   59.824354
smooth   0.02      TPU3_unit_balanced     0.427789       1.0   60.587854
smooth   0.02      TPU4_unit_balanced     0.393352       1.0   61.928872
smooth   0.02      TPU5_unit_balanced     0.378635       1.0   65.208985
smooth   0.02 flat_W1_MLSQU_HEX_Q2_M1     0.156922       1.0    2.365301
smooth   0.05           H2002_skimage     0.012351       1.0    0.834136
smooth   0.05      TPU2_unit_balanced     0.620991       1.0   35.372648
smooth   0.05      TPU3_unit_balanced     0.358317       1.0   36.532737
smooth   0.05      TPU4_unit_balanced     0.268526       1.0   38.019674
smooth   0.05      TPU5_unit_balanced     0.214268       1.0   39.659470
smooth   0.05 flat_W1_MLSQU_HEX_Q2_M1     0.003685       1.0    0.587426
smooth   0.10           H2002_skimage     0.000025       1.0    0.259782
smooth   0.10      TPU2_unit_balanced     0.585508       1.0   15.671327
smooth   0.10      TPU3_unit_balanced     0.198307       1.0   17.143602
smooth   0.10      TPU4_unit_balanced     0.053735       1.0   14.058115
smooth   0.10      TPU5_unit_balanced     0.018152       1.0   10.737600
smooth   0.10 flat_W1_MLSQU_HEX_Q2_M1     0.000025       1.0    0.259782
smooth   0.25           H2002_skimage     0.000000       1.0    0.104349
smooth   0.25      TPU2_unit_balanced     0.484761       1.0    6.478266
smooth   0.25      TPU3_unit_balanced     0.004880       1.0    2.690735
smooth   0.25      TPU4_unit_balanced     0.000025       1.0    0.109109
smooth   0.25      TPU5_unit_balanced     0.000000       1.0    0.104349
smooth   0.25 flat_W1_MLSQU_HEX_Q2_M1     0.000000       1.0    0.104349
smooth   0.50           H2002_skimage     0.000000       1.0    0.058265
smooth   0.50      TPU2_unit_balanced     0.334163       1.0    3.837023
smooth   0.50      TPU3_unit_balanced     0.000000       1.0    0.058265
smooth   0.50      TPU4_unit_balanced     0.000000       1.0    0.058265
smooth   0.50      TPU5_unit_balanced     0.000000       1.0    0.058265
smooth   0.50 flat_W1_MLSQU_HEX_Q2_M1     0.000000       1.0    0.058265
smooth   1.00           H2002_skimage     0.000000       1.0    0.035415
smooth   1.00      TPU2_unit_balanced     0.139243       1.0    2.344973
smooth   1.00      TPU3_unit_balanced     0.000000       1.0    0.035415
smooth   1.00      TPU4_unit_balanced     0.000000       1.0    0.035415
smooth   1.00      TPU5_unit_balanced     0.000000       1.0    0.035415
smooth   1.00 flat_W1_MLSQU_HEX_Q2_M1     0.000000       1.0    0.035415
```
全组合见summary_all.csv、summary_roi.csv；正常区传播见propagation.csv，均按同一噪声重复与a=1配对。完整repeat0相位、全图/patch整数和连续误差见maps与误差图；有限mask独立存储，整数int32无哨兵冲突。其他重复由精确种子与逐频哈希复现。

## 五个问题的直接回答

1. 主方法相对H2002的逐场景/衰减分类如下；依据repeat级2000次配对bootstrap，原始差值与95%区间在paired_comparisons.csv，不能用BFS失效代替与强基线比较。
```
 scene  a_min                 primary     reference         status  high_accuracy_usable  primary_all_error  reference_all_error  primary_patch_error  reference_patch_error  primary_coverage  reference_coverage
 plane   1.00 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
 plane   0.50 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
 plane   0.25 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
 plane   0.10 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage     当前样本中描述性相近                  True       0.000000e+00         3.814697e-06             0.000000               0.000125               1.0                 1.0
 plane   0.05 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage  主方法错误率更低或覆盖更好                 False       9.307861e-05         5.058289e-04             0.003039               0.016517               1.0                 1.0
 plane   0.02 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage           证据不足                 False       4.747772e-03         4.188538e-03             0.154484               0.136771               1.0                 1.0
smooth   1.00 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
smooth   0.50 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
smooth   0.25 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
smooth   0.10 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage     当前样本中描述性相近                  True       7.629395e-07         7.629395e-07             0.000025               0.000025               1.0                 1.0
smooth   0.05 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage  主方法错误率更低或覆盖更好                 False       1.129150e-04         3.784180e-04             0.003685               0.012351               1.0                 1.0
smooth   0.02 flat_W1_MLSQU_HEX_Q2_M1 H2002_skimage           证据不足                 False       4.825592e-03         5.249023e-03             0.156922               0.171315               1.0                 1.0
```
2. 与8/12/16/20帧的等级表见performance_levels.csv。判定同时要求全图±0.001、patch±0.01、覆盖±0.001；高正确率表述额外要求双方全图错误≤0.001、patch≤0.01、覆盖≥0.99。共同明显失效不包装成有效等效。相同零错只叫当前配置共同通过/天花板；20次全成功的试验成功概率双侧精确95%下界约83.16%，逐方法界另存summary_all.csv。
```
 scene  a_min matched_frame_counts  minimum_matching_frames             note
 plane   0.02                   []                      NaN        无高正确率相近协议
 plane   0.05                   []                      NaN        无高正确率相近协议
 plane   0.10                   []                      NaN        无高正确率相近协议
 plane   0.25         [12, 16, 20]                     12.0 仅当前样本/工程容差，非普适等级
 plane   0.50         [12, 16, 20]                     12.0 仅当前样本/工程容差，非普适等级
 plane   1.00         [12, 16, 20]                     12.0 仅当前样本/工程容差，非普适等级
smooth   0.02                   []                      NaN        无高正确率相近协议
smooth   0.05                   []                      NaN        无高正确率相近协议
smooth   0.10                   []                      NaN        无高正确率相近协议
smooth   0.25         [12, 16, 20]                     12.0 仅当前样本/工程容差，非普适等级
smooth   0.50         [12, 16, 20]                     12.0 仅当前样本/工程容差，非普适等级
smooth   1.00         [12, 16, 20]                     12.0 仅当前样本/工程容差，非普适等级
```
若多个协议均饱和，只报告达到同一高正确率相近成绩的最少帧数，不挑最高帧数包装成更高级别。
3. 正常区错误传播与区域化/W1的作用需结合outside计数、W0/W1及全图H2002比较。没有改变区域形状的消融，不单凭当前结果断言六边形形状的独立因果优势；全图接近100%不能替代patch正确。
```
 scene  a_min                  method  N_bad_out  N_missing  error_total
 plane   0.02           H2002_skimage          0          0     0.000000
 plane   0.02              flat_W0_M0        616          0     0.000485
 plane   0.02 flat_W1_MLSQU_HEX_Q2_M0         22          0     0.000017
 plane   0.02 flat_W1_MLSQU_HEX_Q2_M1         22          0     0.000017
 plane   0.02              global_BFS     280373          0     0.220665
 plane   0.05           H2002_skimage          0          0     0.000000
 plane   0.05              flat_W0_M0        188          0     0.000148
 plane   0.05 flat_W1_MLSQU_HEX_Q2_M0          0          0     0.000000
 plane   0.05 flat_W1_MLSQU_HEX_Q2_M1          0          0     0.000000
 plane   0.05              global_BFS     160251          0     0.126124
 plane   0.10           H2002_skimage          0          0     0.000000
 plane   0.10              flat_W0_M0          0          0     0.000000
 plane   0.10 flat_W1_MLSQU_HEX_Q2_M0          0          0     0.000000
 plane   0.10 flat_W1_MLSQU_HEX_Q2_M1          0          0     0.000000
 plane   0.10              global_BFS      28251          0     0.022235
 plane   0.25           H2002_skimage          0          0     0.000000
 plane   0.25              flat_W0_M0          0          0     0.000000
 plane   0.25 flat_W1_MLSQU_HEX_Q2_M0          0          0     0.000000
 plane   0.25 flat_W1_MLSQU_HEX_Q2_M1          0          0     0.000000
 plane   0.25              global_BFS          0          0     0.000000
 plane   0.50           H2002_skimage          0          0     0.000000
 plane   0.50              flat_W0_M0          0          0     0.000000
 plane   0.50 flat_W1_MLSQU_HEX_Q2_M0          0          0     0.000000
 plane   0.50 flat_W1_MLSQU_HEX_Q2_M1          0          0     0.000000
 plane   0.50              global_BFS          0          0     0.000000
 plane   1.00           H2002_skimage          0          0     0.000000
 plane   1.00              flat_W0_M0          0          0     0.000000
 plane   1.00 flat_W1_MLSQU_HEX_Q2_M0          0          0     0.000000
 plane   1.00 flat_W1_MLSQU_HEX_Q2_M1          0          0     0.000000
 plane   1.00              global_BFS          0          0     0.000000
smooth   0.02           H2002_skimage          0          0     0.000000
smooth   0.02              flat_W0_M0        548          0     0.000431
smooth   0.02 flat_W1_MLSQU_HEX_Q2_M0         23          0     0.000018
smooth   0.02 flat_W1_MLSQU_HEX_Q2_M1         23          0     0.000018
smooth   0.02              global_BFS     296056          0     0.233012
smooth   0.05           H2002_skimage          0          0     0.000000
smooth   0.05              flat_W0_M0        142          0     0.000112
smooth   0.05 flat_W1_MLSQU_HEX_Q2_M0          0          0     0.000000
smooth   0.05 flat_W1_MLSQU_HEX_Q2_M1          0          0     0.000000
smooth   0.05              global_BFS     139921          0     0.110125
smooth   0.10           H2002_skimage          0          0     0.000000
smooth   0.10              flat_W0_M0          0          0     0.000000
smooth   0.10 flat_W1_MLSQU_HEX_Q2_M0          0          0     0.000000
smooth   0.10 flat_W1_MLSQU_HEX_Q2_M1          0          0     0.000000
smooth   0.10              global_BFS        657          0     0.000517
smooth   0.25           H2002_skimage          0          0     0.000000
smooth   0.25              flat_W0_M0          0          0     0.000000
smooth   0.25 flat_W1_MLSQU_HEX_Q2_M0          0          0     0.000000
smooth   0.25 flat_W1_MLSQU_HEX_Q2_M1          0          0     0.000000
smooth   0.25              global_BFS          0          0     0.000000
smooth   0.50           H2002_skimage          0          0     0.000000
smooth   0.50              flat_W0_M0          0          0     0.000000
smooth   0.50 flat_W1_MLSQU_HEX_Q2_M0          0          0     0.000000
smooth   0.50 flat_W1_MLSQU_HEX_Q2_M1          0          0     0.000000
smooth   0.50              global_BFS          0          0     0.000000
smooth   1.00           H2002_skimage          0          0     0.000000
smooth   1.00              flat_W0_M0          0          0     0.000000
smooth   1.00 flat_W1_MLSQU_HEX_Q2_M0          0          0     0.000000
smooth   1.00 flat_W1_MLSQU_HEX_Q2_M1          0          0     0.000000
smooth   1.00              global_BFS          0          0     0.000000
```
W0与W1同形状、相同图后端对照可描述局部展开选择的增量，但不能把W1拟合与区域支撑的作用统称为“六边形独立优势”。
在a=0.02的两场景20次重复中，主方法outside错误像素合计45个，H2002为0个；因此不能宣称本方法在所有区域始终更好，也不能用全图平均掩盖弱区失效。a=0.05相对H2002的patch优势与最弱档的证据不足应分别陈述。
4. M1相对M0的完整配对结论如下，任何相同或更差均如实保留；不能把尚未实现的连续置信图后端当成本轮收益。
本轮主方法与M0消融在全图错误、patch错误、覆盖率三个指标的逐条件配对差及区间全部为0：True。若为True，应明确结论是本轮未观察到M1额外收益，不能宣称M1带来改善。
```
 scene  a_min                 primary               reference         status  high_accuracy_usable  primary_all_error  reference_all_error  primary_patch_error  reference_patch_error  primary_coverage  reference_coverage
 plane   1.00 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
 plane   0.50 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
 plane   0.25 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
 plane   0.10 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
 plane   0.05 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0     当前样本中描述性相近                  True       9.307861e-05         9.307861e-05             0.003039               0.003039               1.0                 1.0
 plane   0.02 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0         共同失败区间                 False       4.747772e-03         4.747772e-03             0.154484               0.154484               1.0                 1.0
smooth   1.00 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
smooth   0.50 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
smooth   0.25 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0 当前配置共同通过/天花板结果                  True       0.000000e+00         0.000000e+00             0.000000               0.000000               1.0                 1.0
smooth   0.10 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0     当前样本中描述性相近                  True       7.629395e-07         7.629395e-07             0.000025               0.000025               1.0                 1.0
smooth   0.05 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0     当前样本中描述性相近                  True       1.129150e-04         1.129150e-04             0.003685               0.003685               1.0                 1.0
smooth   0.02 flat_W1_MLSQU_HEX_Q2_M1 flat_W1_MLSQU_HEX_Q2_M0         共同失败区间                 False       4.825592e-03         4.825592e-03             0.156922               0.156922               1.0                 1.0
```
5. 四帧相对8/12/16/20帧分别少50%/66.7%/75%/80%物体帧；代价包括空间连续性、图连通性、绝对锚点、计算开销和弱区可靠性。Python与C实现时间不能直接视为理论速度排名，采集帧减少不等于总耗时同比减少。绝对多频结果独立记录，小数相位噪声对所有方法相同，级次正确也不等于恢复高精度相位。

本轮只验证固定几何/信号机制，未证明双目、远距离、遮挡或任意不连续能力。实拍JPEG未作准确率真值。复现：工作区.venv安装scikit-image==0.25.2，运行`python -m pytest -q . ../E05_frame_equivalence ../E04_real_monocular ../E03_low_modulation ../E02_local_unwrapping ../E01_hex_pairwise --junitxml=outputs/pytest.xml`、`run_e06.py`、`report.py`。实际运行阶段日志run.log，源/依赖版本与历史哈希在input_manifest.json。
