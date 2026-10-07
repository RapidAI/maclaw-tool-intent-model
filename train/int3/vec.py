import numpy as np, json
def load(prefix):
    keys=[l.rstrip('\n') for l in open(prefix+'.keys')]
    V=np.fromfile(prefix+'.f32',dtype='<f4').reshape(len(keys),-1)
    return {k:V[i] for i,k in enumerate(keys)}
