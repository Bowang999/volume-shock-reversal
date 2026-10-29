"""Portable factor definitions, matched-pool controls and evaluation.

Frozen execution functions in engine.py are adapted from the validated research
implementation. No provider-specific paths, credentials, or stock data are used.
"""
from dataclasses import dataclass
from pathlib import Path
from math import erfc,sqrt
import json
import numpy as np
import pandas as pd
import bottleneck as bn
from numba.typed import List
from .engine import ema_strict,rank_only,project,simulate_positive,horizon_ic,performance

MODELS=('base','pure_rev40','control_rev40','control_attention40','joint2')
PERIODS={'research':('2010-01-01','2024-12-31'),'later':('2025-01-01','2026-10-20')}

@dataclass
class Panel:
    """Aligned T-by-N inputs. See data/README.md for precise field semantics."""
    dates:np.ndarray
    close:np.ndarray
    volume:np.ndarray
    float_shares:np.ndarray
    float_cap:np.ndarray
    industry:np.ndarray
    observed:np.ndarray
    tradable:np.ndarray
    eligible:np.ndarray|None=None

    def validate(self):
        shape=self.close.shape
        if len(shape)!=2 or len(self.dates)!=shape[0]:raise ValueError('Expected aligned dates and T-by-N panels')
        for field in ['volume','float_shares','float_cap','industry','observed','tradable']:
            if getattr(self,field).shape!=shape:raise ValueError(f'Misaligned {field}')
        if self.eligible is not None and self.eligible.shape!=shape:raise ValueError('Misaligned eligibility')
        if not np.all(self.dates[1:]>self.dates[:-1]):raise ValueError('Dates must be strictly increasing')
        if np.any(self.tradable & ~self.observed):raise ValueError('Tradable observations need observed quotes')
        finite=np.isfinite(self.close)
        if np.any(finite & (self.close<=0)) or np.any(self.observed & ~finite):
            raise ValueError('Observed and finite marking prices must be positive')
        # NaN before listing is valid; after the first mark, carry the last
        # known close without manufacturing an observed quote or future price.
        after_first=np.maximum.accumulate(finite,axis=0)
        if np.any(after_first & ~finite):
            raise ValueError('Carry last known marks after first observation; keep observed=False for missing quotes')
        return self

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            fields={k:z[k] for k in cls.__dataclass_fields__ if k in z.files}
        fields['dates']=fields['dates'].astype('datetime64[D]')
        for k in ['observed','tradable','eligible']:
            if k in fields:fields[k]=fields[k].astype(bool)
        return cls(**fields).validate()

def shift(a):
    z=np.full(a.shape,np.nan);z[1:]=a[:-1];return z

def raw_signals(panel):
    p=panel.validate();r=np.full(p.close.shape,np.nan)
    good=p.tradable[1:]&p.observed[:-1]&(p.close[:-1]>0)
    r[1:]=np.where(good,p.close[1:]/p.close[:-1]-1,np.nan)
    v=np.where(p.tradable&(p.volume>0)&np.isfinite(p.volume),p.volume,np.nan)
    with np.errstate(invalid='ignore',divide='ignore'):
        r5=np.expm1(bn.move_sum(np.where(r>-1,np.log1p(r),np.nan),5,min_count=5,axis=0))
        v5=shift(bn.move_mean(v,5,min_count=5,axis=0))
        shock=np.where((v>0)&(v5>0),v/v5-1,np.nan)
        tau=np.where(p.tradable&(v>0)&(p.float_shares>0),v/p.float_shares,np.nan)
        tau20=shift(bn.move_mean(tau,20,min_count=20,axis=0))
        attention=np.where((tau>0)&(tau20>0),tau/tau20-1,np.nan)
    return {'factor':ema_strict(-r5*np.maximum(shock,0.),40),
            'reversal':ema_strict(-r5,40),'attention':ema_strict(attention,40)}

def build_signals(panel,min_cross=50,min_industry=3):
    """All scenarios share one current-day pool and exactly one joint OLS.

    Controls are ranks of the raw benchmarks, not pre-neutralized residuals.
    Historical eligible membership is an explicit optional input.
    """
    p=panel;raw=raw_signals(p);mask=p.tradable.copy()
    if p.eligible is not None:mask &= p.eligible
    for a in raw.values():mask &= np.isfinite(a)
    ranks,n=rank_only(raw['factor'],p.float_cap,p.industry,mask,min_industry,min_cross)
    common=np.isfinite(ranks)
    reversal,_=rank_only(raw['reversal'],p.float_cap,p.industry,common,min_industry,min_cross)
    attention,_=rank_only(raw['attention'],p.float_cap,p.industry,common,min_industry,min_cross)
    targets={'base':(ranks,[]),'pure_rev40':(reversal,[]),'control_rev40':(ranks,[reversal]),
             'control_attention40':(ranks,[attention]),'joint2':(ranks,[reversal,attention])}
    scores={}
    for name,(target,controls) in targets.items():
        xx=controls if controls else [np.zeros(p.close.shape)]
        scores[name]=project(target,List([np.asarray(x) for x in xx]),p.float_cap,p.industry,min_cross)
        if not np.array_equal(np.isfinite(scores[name]),common):raise ValueError('Degenerate or unmatched cross section')
    return scores,common

def hac_mean(x,lag=21):
    a=np.asarray(x,float);a=a[np.isfinite(a)];n=len(a)
    if n<3:return dict(n=n,mean=np.nan,se=np.nan,t=np.nan,p_two_sided=np.nan,CI95_low=np.nan,CI95_high=np.nan)
    z=a-a.mean();variance=z@z/n
    for h in range(1,min(lag,n-1)+1):variance+=2*(1-h/(lag+1))*(z[h:]@z[:-h]/n)
    se=np.sqrt(max(variance,0)/n);t=a.mean()/se if se>0 else np.nan
    return dict(n=n,mean=a.mean(),se=se,t=t,p_two_sided=erfc(abs(t)/sqrt(2)) if np.isfinite(t) else np.nan,CI95_low=a.mean()-1.96*se,CI95_high=a.mean()+1.96*se)

def summarize_ic(frame):
    rows=[]
    for (model,period,h),z in frame.groupby(['model','period','horizon'],sort=False):
        for metric in ['ic','rank_ic']:
            x=z[metric].dropna();sd=x.std(ddof=1)
            rows.append(dict(model=model,period=period,horizon=int(h),metric=metric,**hac_mean(x,max(21,int(h))),
                std=sd,ICIR_unannualized=x.mean()/sd if sd>0 else np.nan,positive_fraction=(x>0).mean()))
    return pd.DataFrame(rows)

def run_research(panel,out,periods=None,min_cross=50,min_industry=3):
    """Run fixed formulas; no sign search, parameter tuning, or price lookahead."""
    periods=periods or PERIODS;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=panel;score,common=build_signals(p,min_cross,min_industry)
    ids=np.flatnonzero(common.sum(1)>=min_cross)
    if not len(ids):raise ValueError('No valid cross sections after strict warmup')
    start=min(x[0] for x in periods.values());beg=int(ids[ids>=np.searchsorted(p.dates,np.datetime64(start))][0]);last=len(p.dates)-1
    rows=[];paths=[];ics=[];cfg={'annual_days':252,'risk_free_annual':0.}
    for model,s in score.items():
        cube=simulate_positive(p.close,p.tradable,s,beg,last,min_cross)
        if not np.isfinite(cube).all():raise ValueError('Invalid ledger / depleted NAV')
        for ix,cost in enumerate([0,10]):
            f=pd.DataFrame(cube[ix],columns=['net_return','turnover','long_exposure','short_exposure','frozen_exposure'])
            f['date']=p.dates[beg+1:last+1].astype(str);f['model']=model;f['cost_bps']=cost;f['nav']=(1+f.net_return).cumprod();paths.append(f)
            for period,(lo,hi) in periods.items():
                z=f[f.date.between(lo,hi)]
                if len(z):rows.append(dict(model=model,period=period,cost_bps=cost,**performance(z,cfg),one_way_turnover_pct=z.turnover.mean()*50))
        for period,(lo,hi) in periods.items():
            first=max(beg,int(np.searchsorted(p.dates,np.datetime64(lo))));end=min(last,int(np.searchsorted(p.dates,np.datetime64(hi),side='right')-1))
            if end<=first:continue
            a=horizon_ic(p.close,p.observed,s,first,end,np.array([1,5,21]),min_cross)
            for j,h in enumerate([1,5,21]):ics.append(pd.DataFrame(dict(date=p.dates[first:end+1].astype(str),n=a[:,j,0],ic=a[:,j,1],rank_ic=a[:,j,2],horizon=h,period=period,model=model)))
    metrics=pd.DataFrame(rows);daily=pd.concat(paths,ignore_index=True);ic=pd.concat(ics,ignore_index=True)
    for name,frame in [('metrics',metrics),('portfolio_daily',daily),('ic_daily',ic),('ic_summary',summarize_ic(ic))]:frame.to_csv(out/(name+'.csv'),index=False,float_format='%.12g')
    pd.DataFrame({'date':p.dates.astype(str),'n_eligible':common.sum(1)}).to_csv(out/'coverage.csv',index=False)
    (out/'method.json').write_text(json.dumps({'models':list(MODELS),'costs_bps':[0,10],'periods':periods,'long_target':1,'short_target':-1,'signal_to_execution':'t close -> t+1 close -> first holding return t+1 to t+2','IC_horizons':[1,5,21],'ICIR':'unannualized mean/std; overlapping labels use HAC21','no_RV_control':True},indent=2),'utf-8')
    return metrics
