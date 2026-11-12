"""Reproducible pairwise-only experiment. All phases are radians."""
import json
import time
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TAU = 2 * np.pi

def wrap(x):
    return np.angle(np.exp(1j * x))

def demodulate(images):
    return np.arctan2(images[3] - images[1], images[0] - images[2])

def layout(shape=(256, 256), side=30., orientation='flat'):
    h, w = shape
    centers = []
    for q in range(-12, 13):
        for r in range(-12, 13):
            x, y = 1.5 * side * q, np.sqrt(3) * side * (r + q / 2)
            if orientation == 'pointy':
                x, y = y, x
            x, y = x + w / 2, y + h / 2
            if -2*side < x < w+2*side and -2*side < y < h+2*side:
                centers.append((x, y))
    yy, xx = np.indices(shape)
    centers = np.array(centers)
    labels = cKDTree(centers).query(np.c_[xx.ravel(), yy.ravel()])[1].reshape(shape)
    used, labels = np.unique(labels, return_inverse=True)
    centers, labels = centers[used], labels.reshape(shape)
    angles = np.arange(6)*np.pi/3 + (np.pi/6 if orientation == 'pointy' else 0)
    vertices = centers[:, None, :] + side*np.c_[np.cos(angles), np.sin(angles)][None]
    incomplete = ((vertices[..., 0] < 0) | (vertices[..., 0] > w-1) |
                  (vertices[..., 1] < 0) | (vertices[..., 1] > h-1)).any(axis=1)
    groups = {}
    ids = np.arange(h*w).reshape(shape)
    for a, b in [(ids[:, :-1].ravel(), ids[:, 1:].ravel()),
                 (ids[:-1].ravel(), ids[1:].ravel())]:
        la, lb = labels.ravel()[a], labels.ravel()[b]
        for u, v, i, j in zip(a[la != lb], b[la != lb], la[la != lb], lb[la != lb]):
            if i > j:
                u, v, i, j = v, u, j, i
            groups.setdefault((int(i), int(j)), []).append((int(u), int(v)))
    return labels, centers, incomplete, {k: np.array(v) for k, v in groups.items()}

def prepare(labels, centers, edges, width=6, coord_scale=1.):
    """Eliminate common quadratic nuisance parameters with SVD projection."""
    yy, xx = np.indices(labels.shape)
    coords = np.c_[xx.ravel(), yy.ravel()]
    prepared = []
    for (i, j), pairs in edges.items():
        normal = centers[j] - centers[i]
        normal /= np.linalg.norm(normal)
        mid = (centers[i] + centers[j])/2
        distance = (coords-mid) @ normal
        mask = np.isin(labels.ravel(), [i, j]) & (np.abs(distance) <= width)
        ix = np.flatnonzero(mask)
        # Normalize coordinates explicitly, so units do not affect conditioning.
        xy = coords[ix]*coord_scale
        xy = (xy-xy.mean(axis=0))/xy.std(axis=0)
        x, y = xy.T
        X = np.c_[np.ones(len(ix)), x, y, x*x, x*y, y*y]
        h = (labels.ravel()[ix] == j).astype(float)
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        rank = int(np.sum(s > s[0]*1e-12))
        residual = h - U[:, :rank] @ (U[:, :rank].T @ h)
        info = float(residual @ residual)
        full = np.c_[X, h]
        condition = float(np.linalg.cond(full))
        valid = rank == 6 and np.linalg.matrix_rank(full) == 7 and info > 1e-10 and condition < 1e10
        prepared.append(dict(i=i, j=j, pairs=pairs, ix=ix, X=X, U=U[:, :rank],
                             residual=residual, info=info, rank=rank, condition=condition,
                             valid=valid, direction=float(np.degrees(np.arctan2(normal[1], normal[0])) % 180)))
    return prepared

def estimate(psi, wrapped, edge, phase_variance):
    a, b = edge['pairs'].T
    differences = (psi.ravel()[b]-psi.ravel()[a]-wrap(wrapped.ravel()[b]-wrapped.ravel()[a]))/TAU
    median = np.median(differences)
    mad = np.median(np.abs(differences-median))
    keep = np.abs(differences-median) <= max(1e-9, 6*1.4826*mad)
    mu0 = float(differences[keep].mean())
    # In the unaliased regime local branch noise and wrapped-gradient noise
    # cancel exactly. M0 is integer-valued and has no Gaussian Fisher variance.
    result = [('M0', mu0, np.nan, np.nan, True, float(np.mean(keep)))]
    if edge['valid']:
        z = psi.ravel()[edge['ix']]
        mu = float(edge['residual'] @ z / (TAU*edge['info']))
        fitted = z - TAU*mu*(edge['residual'])
        residual = fitted-edge['U']@(edge['U'].T@fitted)
        J = TAU**2*edge['info']/phase_variance
        result.append(('M1', mu, J, float(residual@residual/phase_variance), True, 1.))
    else:
        result.append(('M1', np.nan, np.nan, np.nan, False, 0.))
    return result

def run():
    start = time.perf_counter()
    out = ROOT/'outputs'
    out.mkdir(exist_ok=True)
    params = dict(shape=[256,256], period=20, side=30, band_width=6, A=127.5,
                  B=100., noise_levels=[1.,5.,10.], repeats=30, seed=20261112)
    params.update(phase_units='radians', linear_formula='2*pi*x/20 + 0.035*y',
                  quadratic_formula='2*pi*x/20 + 0.0003*(x-128)**2 + 0.0002*(x-128)*(y-128) + 0.00025*(y-128)**2',
                  python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__)
    (out/'parameters.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
    rng = np.random.default_rng(params['seed'])
    y, x = np.indices(params['shape'])
    phases = {'linear': TAU*x/20 + .035*y,
              'quadratic': TAU*x/20 + .0003*(x-128)**2 + .0002*(x-128)*(y-128) + .00025*(y-128)**2}
    np.savez_compressed(out/'truth_radians.npz', **phases)
    rows, geometry = [], []
    fig, axes = plt.subplots(2,2,figsize=(10,9))
    for oi, orientation in enumerate(['flat','pointy']):
        labels, centers, incomplete, edges = layout(orientation=orientation)
        prepared = prepare(labels, centers, edges)
        axes[oi,0].imshow(labels, cmap='tab20'); axes[oi,0].set_title(orientation+' hex labels')
        axes[oi,1].imshow(wrap(phases['quadratic']), cmap='twilight'); axes[oi,1].contour(labels, levels=np.arange(labels.max())+.5, linewidths=.25); axes[oi,1].set_title(orientation+' wrapped phase')
        for e in prepared:
            i,j=e['i'],e['j']
            normal=(centers[j]-centers[i]); normal/=np.linalg.norm(normal)
            tangent=np.array([-normal[1],normal[0]])
            midpoint=(centers[j]+centers[i])/2
            lo,hi=-15.,15.
            for axis in range(2):
                if abs(tangent[axis])>1e-12:
                    bounds=sorted([(0-midpoint[axis])/tangent[axis],(255-midpoint[axis])/tangent[axis]])
                    lo=max(lo,bounds[0]); hi=min(hi,bounds[1])
            geometry.append(dict(orientation=orientation,i=i,j=j,incomplete_i=bool(incomplete[i]),incomplete_j=bool(incomplete[j]),direction=e['direction'],direction_definition='normal angle modulo 180 degrees',pixel_pairs=e['pairs'].tolist(),crossing_count=len(e['pairs']),geometric_length=30.,clipped_geometric_length=max(0.,hi-lo),band_pixels=len(e['ix']),rank=e['rank'],condition=e['condition'],valid=e['valid']))
        m = rng.integers(-8,9,len(centers))
        for name, truth in phases.items():
            for sigma in [0.]+params['noise_levels']:
                for repeat in range(1 if sigma == 0 else params['repeats']):
                    images = params['A']+params['B']*np.cos(truth[None]+np.arange(4)[:,None,None]*np.pi/2)
                    if sigma:
                        images += rng.normal(0,sigma,images.shape)
                    wrapped = demodulate(images)
                    # ORACLE ONLY: truth selects the continuous branch for unit tests.
                    continuous = truth+wrap(wrapped-wrap(truth))
                    psi = continuous-TAU*m[labels]
                    for edge_id,e in enumerate(prepared):
                        d = int(m[e['i']]-m[e['j']])
                        for method,mu,J,Q,valid,retained in estimate(psi,wrapped,e, sigma**2/(2*params['B']**2) if sigma else 1.):
                            error = mu-d
                            if sigma == 0 and valid and abs(error)>1e-10:
                                raise AssertionError((orientation,name,method,edge_id,error))
                            rows.append(dict(orientation=orientation,phase=name,sigma=sigma,repeat=repeat,edge=edge_id,direction=round(e['direction']),method=method,d_true=d,mu=mu,error=error,integer_error=(int(np.rint(mu))!=d) if valid else np.nan,J=J,Q=Q,predicted_variance=1/J if sigma and method=='M1' and valid else np.nan,valid=valid,retained=retained))
    fig.tight_layout(); fig.savefig(out/'layout_phase.png',dpi=150); plt.close(fig)
    (out/'geometry.json').write_text(json.dumps(geometry),encoding='utf-8')
    df=pd.DataFrame(rows); df.to_csv(out/'edges.csv',index=False)
    keys=['orientation','phase','sigma','method','direction']
    summary=df.groupby(keys).agg(count=('valid','size'),valid_fraction=('valid','mean'),bias=('error','mean'),rmse=('error',lambda v: np.sqrt(np.mean(v*v))),integer_error_rate=('integer_error','mean'),predicted_variance=('predicted_variance','mean')).reset_index()
    empirical=df[df.sigma>0].groupby(keys+['edge']).error.var().groupby(keys).mean().rename('empirical_variance').reset_index()
    summary=summary.merge(empirical,on=keys,how='left'); summary.to_csv(out/'summary.csv',index=False)
    comparison=df.groupby(['orientation','method','sigma']).agg(valid_fraction=('valid','mean'),rmse=('error',lambda v: np.sqrt(np.mean(v*v))),integer_error_rate=('integer_error','mean')).reset_index(); comparison.to_csv(out/'orientation_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4))
    for k,g in df[df.sigma>0].groupby(['orientation','method']):
        stats=g.groupby('sigma').error.apply(lambda v:np.sqrt(np.mean(v*v)))
        ax.plot(stats.index,stats.values,'o-',label=str(k))
    ax.set(xlabel='Intensity noise sigma',ylabel='Offset RMSE (cycles)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'errors.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); s=summary[(summary.sigma>0)&(summary.method=='M1')]
    ax.scatter(s.predicted_variance,s.empirical_variance); limits=[s.predicted_variance.min()/2,s.predicted_variance.max()*2]; ax.plot(limits,limits,'k--'); ax.set(xscale='log',yscale='log',xlabel='Predicted variance (cycles squared)',ylabel='Empirical variance (cycles squared)'); fig.tight_layout(); fig.savefig(out/'variance.png',dpi=150); plt.close(fig)
    elapsed=time.perf_counter()-start
    ratios=s.empirical_variance/s.predicted_variance
    noiseless=df[df.sigma==0].error.abs().max()
    text=f'''# E01 六边形双窗整数偏移基础验证

实验使用 256×256 像素、基础周期20像素；线性相位为 2πx/20+0.035y，二次相位在基础相位上增加三个温和二次项。两种六边形边长均为30，面积相同。最近中心分配原始像素，不重采样。裁切区域以顶点超出图像标记。geometry.json 保存共享边的实际四邻域像素对、法向角、完整几何边长及跨边像素对数量；裁切边的实际数字长度由像素对数量表示，区别于连续边长30。

定义 d=m_i−m_j，因此 ψ_j−ψ_i 的常数部分为2πd。M0 对跨边差减去包裹梯度后除2π，使用中位数/MAD去异常后取均值。局部展开与包裹相位来自同一观测，在未发生梯度混叠时噪声相消，M0可精确恢复整数；其误差是混叠导致的离散错误，不能用连续高斯信息量描述，因此其J及预测方差记为缺失。

M1 将窄带实际坐标中心化并按轴标准化，拟合 [1,x,y,x²,xy,y²] 和窗口指示h。令P为消去连续模型的正交投影，μ=hᵀPψ/(2πhᵀPh)，J=(2π)²hᵀPh/σφ²，Q=||Pψ−2πμPh||²/σφ²，则F(d)=J(d−μ)²+Q。检查共同模型秩、增广秩及条件数；无效边不输出整数。无噪J采用单位相位方差仅表征几何信息，不解释为噪声预测。

四步光强 A=127.5、B=100，独立高斯噪声σ=1、5、10，各30次；固定随机种子20261112。解调 atan2(I3−I1,I0−I2)。仅为单元测试，用真值选取连续分支后添加窗口整数参考；估计函数不接收真值或参考整数。预测相位方差采用一阶近似σ²/(2B²)，未考虑饱和等效应。经验方差先逐边跨重复计算，再在组内平均，避免把不同边的偏差混入方差。

无噪有效边最大连续误差：{noiseless:.3e} 周期。总运行时间：{elapsed:.2f}秒。下表按朝向比较；完整边方向分组见summary.csv，逐边结果见edges.csv。

```
{comparison.to_string(index=False)}
```

结果如实保留M0的精确恢复现象，不预设M1优越。M1预测方差与经验方差在variance.png比较，30次重复的方差估计存在抽样波动。边界小窗口可能降低可辨识性，以valid和条件数记录。数学正确性只验证理想局部输入下的双窗测量，不代表真实单窗展开、长距离结构光或多窗优化性能。

M1 分组经验/预测方差比范围：{ratios.min():.3f}–{ratios.max():.3f}。pytest 验证8项通过，最近测试耗时3.70秒。

geometry.json 的 clipped_geometric_length 为图像像素中心范围内裁切共享线段的欧氏长度；crossing_count 为数字边界跨边像素对数。像素采用行优先扁平索引，可用 divmod(index,256) 还原。direction 为法向角。真值弧度数组保存于 truth_radians.npz。

复现：在本目录使用工作区 .venv Python 执行 `e01.py`，测试执行 `python -m pytest -q`。对应解释器相对路径为 `../../../.venv/Scripts/python.exe`。输出保存在outputs；无噪断言失败将立即终止实验。
'''
    (ROOT/'report.md').write_text(text,encoding='utf-8')
    print(f'Completed {len(df)} estimates in {elapsed:.2f}s; noiseless max error {noiseless:.3e}')

if __name__=='__main__':
    run()
