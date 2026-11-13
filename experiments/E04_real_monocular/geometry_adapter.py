"""Same regular lattice and historical support, generalized to full image size."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'E01_hex_pairwise'))
from e01_c import support

def full_layout(shape,side=30.,orientation='flat'):
    h,w=shape; extent=int(np.ceil(max(shape)/side))+3
    centers=[]
    for q in range(-extent,extent+1):
        for r in range(-extent,extent+1):
            x,y=1.5*side*q,np.sqrt(3)*side*(r+q/2)
            if orientation=='pointy': x,y=y,x
            x,y=x+w/2,y+h/2
            if -2*side<x<w+2*side and -2*side<y<h+2*side: centers.append((x,y))
    centers=np.asarray(centers); yy,xx=np.indices(shape)
    ids=cKDTree(centers).query(np.c_[xx.ravel(),yy.ravel()])[1]
    used,labels=np.unique(ids,return_inverse=True); labels=labels.reshape(shape); centers=centers[used]
    angle=np.arange(6)*np.pi/3+(np.pi/6 if orientation=='pointy' else 0)
    v=centers[:,None]+side*np.c_[np.cos(angle),np.sin(angle)][None]
    inc=((v[...,0]<0)|(v[...,0]>w-1)|(v[...,1]<0)|(v[...,1]>h-1)).any(axis=1)
    pixel=np.arange(h*w).reshape(shape); groups={}
    for a,b in [(pixel[:,:-1].ravel(),pixel[:,1:].ravel()),(pixel[:-1].ravel(),pixel[1:].ravel())]:
        i,j=labels.ravel()[a],labels.ravel()[b]; keep=i!=j
        for u,v,i,j in zip(a[keep],b[keep],i[keep],j[keep]):
            if i>j: u,v,i,j=v,u,j,i
            groups.setdefault((int(i),int(j)),[]).append((u,v))
    return labels,centers,inc,{k:np.array(v) for k,v in groups.items()}

def local_support(labels,centers,edges,mask,incomplete):
    h,w=labels.shape; result=[]
    for key,pairs in edges.items():
        selected=centers[list(key)]; x0,y0=np.maximum(0,np.floor(selected.min(axis=0)-32)).astype(int)
        x1,y1=np.minimum([w,h],np.ceil(selected.max(axis=0)+33)).astype(int)
        def local(ids):
            yy,xx=np.divmod(ids,w); return (yy-y0)*(x1-x0)+(xx-x0)
        e=support(labels[y0:y1,x0:x1],centers-[x0,y0],{key:local(pairs)},mask[y0:y1,x0:x1],incomplete)[0]
        def global_ids(ids):
            yy,xx=np.divmod(ids,x1-x0); return (yy+y0)*w+xx+x0
        e['ix']=global_ids(e['ix']); e['pairs']=global_ids(e['pairs']); e['mid']+=np.array([x0,y0])
        result.append(e)
    return result
