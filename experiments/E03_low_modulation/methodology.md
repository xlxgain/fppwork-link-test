# 机制、冻结方法和统计限定

[Lv & Qian (2023), Modeling the measurement precision of Fringe Projection Profilometry, DOI 10.1038/s41377-023-01294-0](https://www.nature.com/articles/s41377-023-01294-0)讨论光强到相位再到几何精度的误差传播；本轮仅采用独立等方差四步光强的局部线性近似 v_phi=σ_I²/(2B_true²)。其前提是相位小扰动，不能转换为整数正确概率或极弱信号下的置信保证。

[Yu et al. (2020), DOI 10.1364/OE.398492](https://pubmed.ncbi.nlm.nih.gov/32752442/)指出低调制及不均匀反射率使随机噪声引起的相位误差不可忽略。本项目仅借此确认现实机制，不引入论文的神经网络，也不声称参数来自该设备。

具体衰减幅值、30/50mm斑块、27.5偏置、100直流/幅值、σ=5均为附件预定探索性配置。a同时缩放直接光直流和幅值，固定噪声代表读出/电子噪声主导的简化，不含散粒噪声。浮点值可以超出0–255而不裁剪；不是8-bit完整成像链路。

本目录通过import复用E02 local_unwrap及E01 support/estimate，未复制改写算法。实际执行侧边长30px，support的默认半宽6px/留白6px，W1内部固定8项张量积Q2和40个rho；JSON仅记录这些生效值，无幅值加权或质量筛选。B_hat仅观察，B_true、分区、深度和斑块信息只存在生成或评价代码，不传估计器。

评价T_obs=Φ_true+wrap(phi−wrap(Φ_true))。窗内整数误差的众数并列取最小绝对值、再较小整数；该整体参考只用于评分。有限输出的内部错数N_bad_out直接计数，缺失像素独立计数；总失败率=(N_bad_out+N_missing)/N_all。无输出c=NaN，空几何集合不适用。算法返回但内部错与算法拒绝分开记录。单窗输出原样接双窗，d_reference=c_j−c_i仅在两端严格成功时存在。

M1遗留J/Q仍用单个标量方差；命名legacy_homoscedastic、状态标记uncalibrated。独立评价q=residual/(2πinfo)，v_mu=sum(q² σ_I²/(2B_true²))，不改变点估计或选点。较高信噪比组预定min_support B_true/σ≥5，只是诊断约定；sigma=0方差为0，不除零。oracle逐边重复经验方差和真实方法条件方差分别保存，不把不同边空间偏差混入。

数据252条件但只有40份非零噪声底稿；档位、朝向、算法相互配对相关。汇总先累加分子分母，不平均组比值。观察相位、窗内级次、边连接三个评价层次独立。outside计数沿用同一个窗参考c，按相同scene/repeat/window对照a=1。联合配对成功只代表当前局部单元，不是整图重建成功率。
