"""Only new observation mechanism. Geometry and phase remain frozen."""
import hashlib
import numpy as np

def digest(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def attenuation(depth,intrinsics,center,minimum,inner=30.,outer=50.):
    y,x=np.indices(depth.shape)
    X=depth*(x-intrinsics['cx'])/intrinsics['fc']; Y=depth*(y-intrinsics['cy'])/intrinsics['fc']
    r=np.hypot(X-center[0],Y-center[1])
    w=np.where(r<=inner,1.,np.where(r>=outer,0.,(1+np.cos(np.pi*(r-inner)/(outer-inner)))/2))
    return 1-(1-minimum)*w,np.where(r<=inner,0,np.where(r>=outer,2,1))

def noise_field(shape,seed,source_scene_index,sigma,repeat):
    # Preserve E02 normal(0, sigma) operation for bitwise a=1 regression.
    rng=np.random.default_rng(np.random.SeedSequence([seed,source_scene_index,int(sigma),repeat]))
    return rng.normal(0,sigma,(4,*shape))

def intensities(truth,a,noise):
    cosine=np.cos(truth[None]+np.arange(4)[:,None,None]*np.pi/2)
    # Separate terms give EXACT E02 evaluation order when a=1.
    return (27.5+100*a)[None]+(100*a)[None]*cosine+noise

def amplitude(images): return .5*np.hypot(images[0]-images[2],images[3]-images[1])
