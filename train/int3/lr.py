import numpy as np
def auc(s,y):
    s=np.asarray(s,float); y=np.asarray(y,bool); pos=s[y]; neg=s[~y]
    return ((pos[:,None]>neg[None,:]).sum()+0.5*(pos[:,None]==neg[None,:]).sum())/(len(pos)*len(neg))
def fit(X,y,lam=1.0,iters=1500,lr=0.5):
    """L2-regularised logistic regression (mean log-loss + lam/2n ||w||^2), Nesterov GD. Returns w,b."""
    n,d=X.shape; y=y.astype(float); w=np.zeros(d); b=0.0; vw=np.zeros(d); vb=0.0
    L=0.25*np.linalg.norm(X,2)**2/n+lam/n; step=1.0/L
    for t in range(iters):
        wl=w+0.9*vw; bl=b+0.9*vb
        p=1/(1+np.exp(-(X@wl+bl))); g=X.T@(p-y)/n+lam/n*wl; gb=(p-y).mean()
        vw=0.9*vw-step*g; vb=0.9*vb-step*gb; w=w+vw; b=b+vb
    return w,b
def cv(X,y,lam,k=5,seed=0):
    rng=np.random.default_rng(seed); idx=np.arange(len(y)); rng.shuffle(idx); folds=np.array_split(idx,k); s=np.zeros(len(y))
    for f in folds:
        tr=np.setdiff1d(idx,f); w,b=fit(X[tr],y[tr],lam); s[f]=X[f]@w+b
    return auc(s,y), s
