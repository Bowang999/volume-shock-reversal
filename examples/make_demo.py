"""Synthetic demonstration ONLY: these data do not generate the empirical figures."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def make(path,days=640,stocks=120):
    rng=np.random.default_rng(554040);dates=pd.bdate_range('2023-06-01',periods=days).to_numpy(dtype='datetime64[D]')
    r=rng.normal(0,.017,(days,stocks))+rng.normal(0,.005,(days,1));close=10*np.exp(np.cumsum(r,axis=0))
    shares=np.broadcast_to(rng.lognormal(18,.7,stocks),(days,stocks)).copy();volume=rng.lognormal(13,.7,(days,stocks));cap=shares*close
    industry=np.broadcast_to(np.arange(stocks)%8,(days,stocks)).copy();observed=np.ones((days,stocks),bool);tradable=observed.copy()
    tradable[260:265,0]=False;volume[260:265,0]=0;close[260:265,0]=close[259,0]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,dates=dates,close=close,volume=volume,float_shares=shares,float_cap=cap,industry=industry,observed=observed,tradable=tradable)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='data/local/synthetic.npz');a=p.parse_args();print(make(a.out))
