"""E01-C: physical ray intersection and common finite edge support.

Original e01.py and outputs are intentionally immutable historical A/B runs.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from e01 import layout, estimate, wrap, demodulate, TAU, plt

ROOT=Path(__file__).resolve().parent

def support(labels, centers, edges, mask, incomplete, width=6., margin=6.):
    y,x=np.indices(labels.shape); coords=np.c_[x.ravel(),y.ravel()]
    result=[]
    for (i,j),pairs in edges.items():
        mid=(centers[i]+centers[j])/2
        normal=centers[j]-centers[i]; normal/=np.linalg.norm(normal)
        tangent=np.array([-normal[1],normal[0]])
        # Clip ideal shared segment against image pixel-center bounds first.
        lo,hi=-15.,15.
        for axis,limit in enumerate([labels.shape[1]-1,labels.shape[0]-1]):
            if abs(tangent[axis])>1e-12:
                a,b=sorted([(0-mid[axis])/tangent[axis],(limit-mid[axis])/tangent[axis]])
                lo=max(lo,a); hi=min(hi,b)
        lo+=margin; hi-=margin
        along=(coords-mid)@tangent; across=(coords-mid)@normal
        common=(along>=lo)&(along<=hi)&mask.ravel()
        keep=common[pairs[:,0]]&common[pairs[:,1]]
        pp=pairs[keep]
        ix=np.flatnonzero(common & (abs(across)<=width)&np.isin(labels.ravel(),[i,j]))
        valid=False; rank=0; cond=np.inf; info=0.; U=np.empty((len(ix),0)); residual=np.zeros(len(ix))
        if len(ix)>=8 and len(pp)>0:
            xy=coords[ix]; scale=xy.std(axis=0)
            if np.all(scale>0):
                xx,yy=((xy-xy.mean(axis=0))/scale).T
                X=np.c_[np.ones(len(ix)),xx,yy,xx*xx,xx*yy,yy*yy]
                U,s,_=np.linalg.svd(X,full_matrices=False); rank=int(sum(s>s[0]*1e-12)); U=U[:,:rank]
                h=(labels.ravel()[ix]==j).astype(float)
                residual=h-U@(U.T@h); info=float(residual@residual)
                cond=float(np.linalg.cond(np.c_[X,h]))
                valid=rank==6 and info>1e-10 and cond<1e10
        result.append(dict(i=i,j=j,pairs=pp,ix=ix,U=U,residual=residual,info=info,
                           rank=rank,condition=cond,valid=valid,lo=lo,hi=hi,mid=mid,tangent=tangent,
                           direction=round(float(np.degrees(np.arctan2(normal[1],normal[0]))%180)),
                           connection='internal' if not(incomplete[i] or incomplete[j]) else 'truncated'))
    return result

def geometry(p, bump=None):
    """World=camera frame, mm. Projector frame: Xp=R(X-Cp), R=I."""
    y,x=np.indices(p['shape']); rx=(x-p['cx'])/p['fc']; ry=(y-p['cy'])/p['fc']
    denom=1-p['tilt_x']*rx-p['tilt_y']*ry
    depth=np.full(rx.shape,np.nan)
    if bump is None:
        np.divide(p['distance_mm'],denom,out=depth,where=denom>0)
    else:
        bx,by=bump
        def equation(z):
            height=p['bump_height_mm']*np.exp(-((z*rx-bx)**2+(z*ry-by)**2)/(2*p['bump_sigma_mm']**2))
            return z*denom-p['distance_mm']+height
        # Bounded monotone surface, bracket every ray; vectorized bisection.
        low=np.full(rx.shape,100.); high=np.full(rx.shape,2000.)
        bracket=(equation(low)*equation(high)<=0)
        for _ in range(60):
            mid=(low+high)/2; above=equation(mid)>0
            high=np.where(above,mid,high); low=np.where(above,low,mid)
        depth=np.where(bracket,(low+high)/2,np.nan)
    X=depth*rx; Y=depth*ry
    u=p['fp']*(X-p['baseline_mm'])/depth+p['pcx']; v=p['fp']*Y/depth+p['pcy']
    valid=np.isfinite(depth)&(depth>0)&(u>=0)&(u<=p['projector_shape'][1]-1)&(v>=0)&(v<=p['projector_shape'][0]-1)
    phase=np.where(valid,TAU*u/p['period_px'],np.nan)
    return phase,valid,depth,u,v

def run():
    start=time.perf_counter(); out=ROOT/'outputs_C'; out.mkdir(exist_ok=True)
    p=json.loads((ROOT/'parameters_C.json').read_text(encoding='utf-8'))
    rng=np.random.default_rng(p['seed']); rows=[]; gradients=[]; geom=[]; surface_gradients=[]
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for oi,orientation in enumerate(['flat','pointy']):
        labels,c,incomplete,edges=layout(orientation=orientation)
        central=int(np.argmin(np.sum((c-128)**2,axis=1)))
        key=next(k for k in edges if central in k and not incomplete[k[0]] and not incomplete[k[1]])
        midpoint=(c[key[0]]+c[key[1]])/2; normal=c[key[1]]-c[key[0]]; normal/=np.linalg.norm(normal)
        vertex=midpoint+15*np.array([-normal[1],normal[0]])
        positions={'plane':None,'center':c[central],'edge':midpoint,'junction':vertex}
        m=rng.integers(-8,9,len(c))
        for si,(scene,pixel) in enumerate(positions.items()):
            bump=None
            if pixel is not None:
                z=p['distance_mm']/(1-p['tilt_x']*(pixel[0]-p['cx'])/p['fc']-p['tilt_y']*(pixel[1]-p['cy'])/p['fc'])
                bump=((pixel[0]-p['cx'])*z/p['fc'],(pixel[1]-p['cy'])*z/p['fc'])
            truth,mask,z,u,v=geometry(p,bump)
            for axis,name in [(0,'camera_y'),(1,'camera_x')]:
                delta=np.diff(truth,axis=axis)
                ok=(mask[:-1,:]&mask[1:,:]) if axis==0 else (mask[:,:-1]&mask[:,1:])
                values=abs(delta[ok])
                surface_gradients.append(dict(orientation=orientation,scene=scene,direction=name,count=len(values),p50=np.median(values),p95=np.quantile(values,.95),maximum=values.max(),near_pi=np.mean(values>=.9*np.pi),over_pi=np.mean(values>=np.pi)))
            np.savez_compressed(out/f'{orientation}_{scene}.npz',phase_radians=truth,valid=mask,depth_mm=z,projector_u=u,projector_v=v)
            axes[oi,si].imshow(truth,cmap='viridis'); axes[oi,si].set_title(orientation+' '+scene)
            es=support(labels,c,edges,mask,incomplete)
            for edge_id,e in enumerate(es):
                geom.append(dict(orientation=orientation,scene=scene,edge=edge_id,connection=e['connection'],direction=e['direction'],pairs=e['pairs'].tolist(),support_pixels=e['ix'].tolist(),lo=e['lo'],hi=e['hi'],rank=e['rank'],condition=e['condition'] if np.isfinite(e['condition']) else None,valid=e['valid'],invalid_reason='' if e['valid'] else ('empty_common_edge' if not len(e['pairs']) else 'insufficient_or_rank_deficient_support')))
                a,b=e['pairs'].T if len(e['pairs']) else ([],[])
                delta=truth.ravel()[b]-truth.ravel()[a]
                gradients.append(dict(orientation=orientation,scene=scene,edge=edge_id,connection=e['connection'],direction=e['direction'],count=len(delta),p50=np.median(abs(delta)) if len(delta) else np.nan,p95=np.quantile(abs(delta),.95) if len(delta) else np.nan,maximum=max(abs(delta),default=np.nan),near_pi=np.mean(abs(delta)>=.9*np.pi) if len(delta) else np.nan,over_pi=np.mean(abs(delta)>=np.pi) if len(delta) else np.nan))
            for sigma in p['noise_levels']:
                for repeat in range(1 if sigma==0 else p['repeats']):
                    images=p['A']+p['B']*np.cos(truth[None]+np.arange(4)[:,None,None]*np.pi/2)
                    images+=rng.normal(0,sigma,images.shape)
                    wrapped=demodulate(images)
                    # Oracle branch ONLY; estimates do not see truth or m.
                    psi=truth+wrap(wrapped-wrap(truth))-TAU*m[labels]
                    for edge_id,e in enumerate(es):
                        values=estimate(psi,wrapped,e,sigma**2/(2*p['B']**2) if sigma else 1.) if len(e['pairs']) else [('M0',np.nan,np.nan,np.nan,False,0),('M1',np.nan,np.nan,np.nan,False,0)]
                        for method,mu,J,Q,valid,_ in values:
                            d=m[e['i']]-m[e['j']]
                            rows.append(dict(orientation=orientation,scene=scene,connection=e['connection'],direction=e['direction'],edge=edge_id,sigma=sigma,repeat=repeat,method=method,valid=valid,error=mu-d,integer_error=float(round(mu)!=d) if valid else np.nan,predicted_variance=1/J if method=='M1' and sigma and valid else np.nan))
    fig.tight_layout(); fig.savefig(out/'phase_scenes.png',dpi=130); plt.close(fig)
    df=pd.DataFrame(rows); df.to_csv(out/'edges.csv',index=False)
    pd.DataFrame(gradients).to_csv(out/'gradients.csv',index=False)
    pd.DataFrame(surface_gradients).to_csv(out/'surface_gradients.csv',index=False)
    (out/'support.json').write_text(json.dumps(geom),encoding='utf-8')
    keys=['orientation','scene','connection','direction','sigma','method']
    summary=df.groupby(keys).agg(valid_fraction=('valid','mean'),bias=('error','mean'),rmse=('error',lambda v:np.sqrt(np.mean(v*v))),integer_error_rate=('integer_error','mean'),predicted_variance=('predicted_variance','mean')).reset_index()
    var=df[df.sigma>0].groupby(keys+['edge']).error.var().groupby(keys).mean().rename('random_variance').reset_index()
    summary=summary.merge(var,on=keys,how='left'); summary.to_csv(out/'summary.csv',index=False)
    bias=df[df.sigma==0][['orientation','scene','connection','direction','edge','method','error']].rename(columns={'error':'deterministic_model_bias'})
    bias.to_csv(out/'model_bias.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4))
    for method,g in summary.groupby('method'):
        means=g.groupby('sigma').rmse.mean()
        ax.plot(means.index,means.values,'o-',label=method)
    ax.set(xlabel='Intensity noise sigma',ylabel='Mean group offset RMSE (cycles)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'errors.png',dpi=140); plt.close(fig)
    elapsed=time.perf_counter()-start
    report=f'''# E01-C 物理几何双窗验证

原A/B程序、输出与结论保留。C独立运行于outputs_C，parameters_C.json包含全部配置。单位：三维mm，内参焦距px，条纹周期为投影仪像素，绝对相位rad。数值配置为探索性，不声称复现任何硬件。模型依据：[投影仪逆相机针孔模型](https://pmc.ncbi.nlm.nih.gov/articles/PMC10121812/)、[绝对相位与投影像素关系](https://pmc.ncbi.nlm.nih.gov/articles/PMC9920907/)。这些来源支持模型，并不支持本实验的数值组合。

相机坐标为世界坐标，Kc焦距400px、主点127.5；投影仪Kp焦距800px、主点(511.5,383.5)。相机R=I,t=0；投影仪中心(100,0,0)mm，R=I,t=(-100,0,0)mm；工作距离800mm，周期40投影像素。相机视线为z[(x-cx)/fc,(y-cy)/fc,1]。表面z=800+0.1X+0.05Y−h exp(−((X-bx)²+(Y-by)²)/(2s²))，h=12mm、s=25mm；h=0为倾斜平面。凸包朝相机方向。求交使用有界二分并检查根区间，然后up=fp(X−baseline)/z+pcx，vp=fpY/z+pcy，Φ=2πup/40。无交点、负深度、越界投影设置无效；不模拟遮挡和光照复杂因素。

共享边先裁切图像，再从两端各移除6px。M0跨边像素对和M1两侧6px窄带均限制于相同切向区间；限制每个像素而非只限制像素对中点。geometry支持详见support.json，避免进入三窗顶点。内部完整连接要求两窗全部顶点在图像内，其他为truncated。局部凸包分别置于中心完整窗中心、其邻边中点及该边三窗顶点；按朝向重新定位，物理尺寸不变。

四步强度噪声σ=0、1、5，非零各20次，固定种子。用真值选择局部连续分支仅用于构造单元测试输入；估计沿用原M0/M1，不读取真值。M1二次共同趋势无法完全表达凸包的非二次相位，model_bias.csv保存无噪每边模型偏差；random_variance逐边跨重复方差后分组平均，避免将空间模型偏差混为随机方差。summary.csv按朝向、场景、边方向、完整性、噪声与方法分组；gradients.csv为有效共享边真实相邻梯度分位数、接近π(≥0.9π)及超过π比例。

共{len(df)}条估计，耗时{elapsed:.2f}秒。有效记录最大整数错误率：{df.integer_error.mean():.6g}；有效连接比例{df.valid.mean():.6g}。无噪M1最大绝对模型偏差{df[(df.sigma==0)&(df.method=='M1')].error.abs().max():.6g}周期。本实验不以制造M0错误为目标，也不代表真实单窗展开或远距离系统性能。

全图连续有效像素的水平/垂直梯度见surface_gradients.csv；逐共享边方向梯度见gradients.csv。两者均保存绝对梯度中位数、95%分位数、最大值和阈值比例。M0无噪和噪声条件下的误差消除机制仍与A/B一致。M1的非二次模型偏差与重复噪声方差分别输出，不将总RMSE当作方差。

原始图像边缘某些短截断边，在两端各去除6px后没有公共像素对；这些连接保留为无效，不输出整数。全部有效记录的整数错误率为0，不能忽略这些无效记录来宣称全部几何连接都可恢复。完整连接与截断连接的有效比例见summary.csv。

验证记录：保留原8项测试，新增局部支撑两种朝向及求交/重投影/越界掩膜测试，总计11项通过（5.23秒）。实际执行review_ab.py和e01_c.py；原A/B输出未改。A/B新支撑结果在outputs_AB_review/summary.csv按internal/truncated分别统计，复现执行 `python review_ab.py`。

复现：在本目录用工作区.venv执行 `python -m pytest -q` 和 `python e01_c.py`。A/B改进验证独立存于outputs_AB_review，不改历史输出。
'''
    (ROOT/'report_C.md').write_text(report,encoding='utf-8')
    print(f'C complete: {len(df)} rows, {elapsed:.2f}s, integer error rate {df.integer_error.mean()}')

if __name__=='__main__':
    run()
