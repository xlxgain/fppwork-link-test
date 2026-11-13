# E02 无真值单窗展开与双窗接口验证

实际基线为63982bfdc982ee75c2a724c245067f61782352d0，新分支experiment/E02-local-unwrapping。历史E01文件未变。固定flat参考四场景，物理位置和原几何参数记录于input_manifest.json；pointy中的center/edge/junction只是场景名，不声称仍对应其网格位置。每个(scene,sigma,repeat)只生成一次四步强度，SHA-256经核对在两朝向、所有算法及oracle对照一致；随机流独立于朝向和窗口数。噪声0/1/5，非零各20次。强度A=127.5、B=100为未饱和、未量化的浮点探索性仿真，不等同8-bit相机链路。

方法与文献差异详见methodology.md。W0确定性BFS；W1张量积Q2拟合加40点rho搜索；均不读取真值，仅访问窗内观测。T_obs只供评价及独立oracle_control。输出原样送入历史M0/M1，禁止真值对齐后输入。模2π同余只证明保留观测，不证明整数正确。

实际pytest：24项，失败/错误0项，用时4.847秒，原始日志outputs_E02/pytest.xml。完整实验耗时257.21秒，逐窗25584条，真实单窗逐边120704条，oracle逐边60352条。每个几何边均保留，含无效短边；覆盖率分母检查通过。最大重包裹误差1.133e-15rad。

```
                 windows  strict_success  coverage  integer_error  continuous_rmse
window_method                                                                     
W0                 12792             1.0       1.0            0.0         0.020703
W1_MLSQU_HEX_Q2    12792             1.0       1.0            0.0         0.020703
```

真实单窗严格成功率1，失败窗口0，两端内部正确且算法有效时双窗整数错误率0。像素加权内部错误、完整/截断窗口与边方向分组见summary_windows.csv、summary_pairs.csv和overall_summary_pairs.csv。严格成功要求单连通、完整输出、内部整数残差全零；失败像素也计入覆盖率和加权错误分母。连续RMSE在评价器只减整体整数参考后单独计算，不能解释为去噪。

当前温和物理配置下接口验证通过；若所有方法零整数错误，不支持宣称W1/M1或某朝向更优。无效截断边仍在分母。fixed_edge_error_variance先固定边跨重复计算再分组，避免模型偏差混入。两端内部不一致时唯一参考缺失，mu/J/Q仅作诊断，见consistency_diagnostics.csv；1/J不是有分支错误时完整算法的无条件方差。

C配对复核在outputs_C_review：冻结物理场景NPZ、oracle逐边完整字段与分组统计。真实单窗结果在outputs_E02：windows.csv、pairs.csv、输入清单、汇总及示例NPZ/图。未加入多窗协调或困难场景，未以制造优势为目标调参。

复现（本目录）：工作区解释器 `../../../.venv/Scripts/python.exe`；依次执行 `-m pytest -q . ../E01_hex_pairwise --junitxml=outputs_E02/pytest.xml`、`run_e02.py`、`finalize.py`。历史几何文件是冻结输入依赖。运行时间含本机实现开销，不用于算法速度排名。
