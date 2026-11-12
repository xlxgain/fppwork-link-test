"""Re-run A/B with finite common support; preserve original outputs."""
import json
import numpy as np
import pandas as pd
from e01 import layout,TAU,wrap,demodulate,estimate
from e01_c import ROOT,support

def run():
    out=ROOT/'outputs_AB_review'; out.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261112); rows=[]
    y,x=np.indices((256,256))
    phases={'linear':TAU*x/20+.035*y,'quadratic':TAU*x/20+.0003*(x-128)**2+.0002*(x-128)*(y-128)+.00025*(y-128)**2}
    for orientation in ['flat','pointy']:
        labels,c,inc,edges=layout(orientation=orientation)
        es=support(labels,c,edges,np.ones(labels.shape,bool),inc)
        m=rng.integers(-8,9,len(c))
        for name,truth in phases.items():
            for sigma in [0,1,5,10]:
                for repeat in range(1 if sigma==0 else 30):
                    images=127.5+100*np.cos(truth[None]+np.arange(4)[:,None,None]*np.pi/2)
                    images+=rng.normal(0,sigma,images.shape)
                    w=demodulate(images); psi=truth+wrap(w-wrap(truth))-TAU*m[labels]
                    for edge,e in enumerate(es):
                        if not len(e['pairs']): continue
                        for method,mu,J,Q,valid,_ in estimate(psi,w,e,sigma*sigma/20000 if sigma else 1):
                            d=m[e['i']]-m[e['j']]
                            if sigma==0 and valid: assert abs(mu-d)<1e-10
                            rows.append(dict(orientation=orientation,phase=name,sigma=sigma,repeat=repeat,edge=edge,connection=e['connection'],direction=e['direction'],method=method,valid=valid,error=mu-d,integer_error=float(round(mu)!=d) if valid else np.nan))
    df=pd.DataFrame(rows); df.to_csv(out/'edges.csv',index=False)
    df.groupby(['orientation','phase','sigma','connection','direction','method']).agg(valid_fraction=('valid','mean'),bias=('error','mean'),rmse=('error',lambda v:np.sqrt(np.mean(v*v))),integer_error_rate=('integer_error','mean')).to_csv(out/'summary.csv')
    (out/'parameters.json').write_text(json.dumps(dict(seed=20261112,noise=[0,1,5,10],repeats=30,width=6,margin=6,source='original A/B formulas; finite common edge support')))
    print('A/B finite-support review complete',len(df))

if __name__=='__main__': run()
