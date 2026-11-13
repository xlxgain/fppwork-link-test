# E06 来源、实现归属与固定条件

[Herráez et al. 2002, Applied Optics 41(35), 7437–7444, DOI 10.1364/AO.41.007437](https://doi.org/10.1364/AO.41.007437)的强基线采用[scikit-image 0.25.2 官方二维unwrap_phase](https://scikit-image.org/docs/0.25.x/api/skimage.restoration.html#skimage.restoration.unwrap_phase)，不是自行重写作者程序。接口(image,wrap_around=False,rng=None)，rng固定0。图像确定性变为[-π,π)，masked array的无效像素不引导展开；输出核验模2π后仅消除浮点数值差。不同输入四连通分量独立调用，屏蔽间隙不桥接。

[Minliang Zhang et al. 2017, DOI 10.1364/OE.25.020381](https://doi.org/10.1364/OE.25.020381)讨论相邻频率比、相移步数与噪声对层级展开的影响。本轮1→60的四个近似等比设计是任务预定配置，不是原文参数或全局最优，也不等同外差/CRT/2-2-3少帧方案。所有频率每频四步，1频在已知0≤u_p<1024和投影相位零点下用正主值初始化；不靠真值修复低信噪分支。

[Lv and Qian Kemao 2023, DOI 10.1038/s41377-023-01294-0](https://doi.org/10.1038/s41377-023-01294-0)支持强度到相位噪声的机制，但弱信号不承诺小扰动高斯模型。沿E03定义同时衰减直接光直流和调制，固定27.5项是非回波缩放的背景/偏置，不代表完整环境光。模型不量化、不裁剪、无额外MTF；数值是探索性配置，不是实测反射率或真实工作距离亮度。

E05的simulated_geometry提供plane与smooth，E03的attenuation提供固定XY斑块。历史最高频周期不同，本轮统一投影全宽1024上的f=60。E05 hex_outputs/spatial_bfs/integer_sync/hierarchical按原函数调用：边长30、支撑半宽6、端点6、W1八项/40rho，图仍round(mu)等权L1。无新增选点、检测器、失败回退或真值权重。

四帧接口只收phi60与mask，不接收低频、幅值、斑块或评分；TPU仅得到自己频率字典。所有方法最高频哈希相同，sigma5每频40份底稿在衰减档复用。a=1使用E05相同SeedSequence并核对哈希/整图输出共同零点，未逐窗对齐。

评价先按输入mask的连通分量选唯一整数众数，再对all/core/transition/outside/patch切分。缺失直接计失败；不按算法图碎片、窗口或ROI重选零点。未锚四帧绝对级次不可识别，原始相位零点仅诊断。单个(128,128)锚点仅是仿真真值模拟外部校准的附表，非免费实测信息。

重复层配对bootstrap种子16、2000次，95%区间；全图±0.001、patch±0.01、覆盖±0.001为预定工程描述标准。高正确率标志需双方全图≤0.001、patch≤0.01、覆盖≥0.99。天花板结果不作正式等效推断，另给试验级精确二项区间；共同明显失效不称高正确率等效。

依赖只定向新增scikit-image==0.25.2及其缺少的依赖，没有无差别升级现有NumPy/SciPy。默认镜像未提供该版本，实际使用官方PyPI安装。实际版本与签名在input_manifest.json，不把Python/C实现时间直接当理论复杂度排名。
