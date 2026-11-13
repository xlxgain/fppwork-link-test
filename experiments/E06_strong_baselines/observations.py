import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; BASE=ROOT.parent
sys.path.insert(0,str(BASE/'E05_frame_equivalence')); sys.path.insert(0,str(BASE/'E03_low_modulation'))
from run_e05 import simulated_geometry
from radiometry import attenuation,digest
from e01 import demodulate

def prepare(scene,cfg):
    geometry=json.loads((BASE/'E01_hex_pairwise/parameters_C.json').read_text())
    z,u,mask=simulated_geometry(scene,geometry,40)
    old=json.loads((BASE/'E03_low_modulation/outputs/input_manifest.json').read_text())
    return z,u,mask,geometry,old['center_XY_mm']

def generate(u,depth,geometry,center,scene,minimum,sigma,repeat,cfg):
    a,zones=attenuation(depth,geometry,center,minimum,cfg['inner_mm'],cfg['outer_mm'])
    phases={}; records=[]
    for f in cfg['frequency_pool']:
        noise=np.random.default_rng(np.random.SeedSequence([cfg['seed'],cfg['scene_indices'][scene],int(sigma),repeat,f])).normal(0,sigma,(4,*u.shape))
        phase=2*np.pi*f*u/1024
        images=(cfg['I_offset']+cfg['A_direct']*a)[None]+(cfg['B_direct']*a)[None]*np.cos(phase[None]+np.arange(4)[:,None,None]*np.pi/2)+noise
        phases[f]=demodulate(images)
        records.append(dict(frequency=f,input_hash=digest(images),noise_hash=digest(noise),outside_0_255=int(np.sum((images<0)|(images>255))),intensity_min=float(images.min()),intensity_max=float(images.max())))
    return phases,zones,a,records
