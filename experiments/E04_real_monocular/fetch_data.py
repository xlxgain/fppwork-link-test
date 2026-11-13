"""Fetch ignored external source; never stage its images."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    cfg=json.loads((ROOT/'config.json').read_text()); cache=ROOT/'data_cache'
    if not cache.exists(): subprocess.run(['git','clone','--depth','1',cfg['source']+'.git',str(cache)],check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cache,text=True).strip()
    if sha!=cfg['source_commit']:
        if subprocess.check_output(['git','status','--porcelain'],cwd=cache,text=True).strip(): raise RuntimeError('External cache has modifications; preserving them')
        subprocess.run(['git','fetch','origin',cfg['source_commit']],cwd=cache,check=True)
        subprocess.run(['git','checkout','--detach',cfg['source_commit']],cwd=cache,check=True)
    print('External data pinned at',cfg['source_commit'])
if __name__=='__main__': main()
