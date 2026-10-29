"""Portable numerical core extracted from the independently validated research implementation.
No data-provider I/O, account information, or local paths.
"""
import numpy as np
import pandas as pd
from numba import njit,prange

@njit(cache=True)
def average_ranks(x):
    order=np.argsort(x);rank=np.empty(len(x),np.float64);start=0
    while start<len(x):
        stop=start+1
        while stop<len(x) and x[order[stop]]==x[order[start]]:stop+=1
        value=(start+1+stop)/2.
        for k in range(start,stop):rank[order[k]]=value
        start=stop
    return rank


@njit(cache=True)
def corr(x,y):
    if len(x)<3:return np.nan
    mx=x.mean();my=y.mean();xx=0.;yy=0.;xy=0.
    for i in range(len(x)):
        a=x[i]-mx;b=y[i]-my;xx+=a*a;yy+=b*b;xy+=a*b
    return xy/np.sqrt(xx*yy) if xx>1e-30 and yy>1e-30 else np.nan


@njit(cache=True,parallel=True)
def preprocess_panel(raw,cap,industry,eligible,min_industry,min_cross):
    rows,cols=raw.shape;out=np.full((rows,cols),np.nan);diag=np.full((rows,4),np.nan)
    for t in prange(rows):
        maxgroup=0
        for j in range(cols):
            if industry[t,j]>maxgroup:maxgroup=industry[t,j]
        counts=np.zeros(maxgroup+1,np.int64);raw_count=0
        for j in range(cols):
            if eligible[t,j] and np.isfinite(raw[t,j]):
                raw_count+=1
                if np.isfinite(cap[t,j]) and cap[t,j]>0 and industry[t,j]>=0:counts[industry[t,j]]+=1
        ids=np.empty(cols,np.int64);n=0
        for j in range(cols):
            if eligible[t,j] and np.isfinite(raw[t,j]) and np.isfinite(cap[t,j]) and cap[t,j]>0 and industry[t,j]>=0 and counts[industry[t,j]]>=min_industry:ids[n]=j;n+=1
        diag[t,0]=raw_count;diag[t,1]=n
        if n<min_cross:continue
        ids=ids[:n];vals=np.empty(n);x=np.empty(n)
        for k in range(n):vals[k]=raw[t,ids[k]];x[k]=np.log(cap[t,ids[k]])
        y=average_ranks(vals);sx=np.zeros(maxgroup+1);sy=np.zeros(maxgroup+1)
        for k in range(n):g=industry[t,ids[k]];sx[g]+=x[k];sy[g]+=y[k]
        xd=np.empty(n);yd=np.empty(n);den=0.;num=0.
        for k in range(n):
            g=industry[t,ids[k]];xd[k]=x[k]-sx[g]/counts[g];yd[k]=y[k]-sy[g]/counts[g];den+=xd[k]*xd[k];num+=xd[k]*yd[k]
        beta=num/den if den>1e-20 else 0.;res=yd-beta*xd;mean=res.mean();var=0.
        for k in range(n):var+=(res[k]-mean)**2
        sd=np.sqrt(var/n)
        if not np.isfinite(sd) or sd<1e-12:diag[t,1]=0;continue
        z=(res-mean)/sd;sums=np.zeros(maxgroup+1)
        for k in range(n):out[t,ids[k]]=z[k];sums[industry[t,ids[k]]]+=z[k]
        error=0.
        for g in range(maxgroup+1):
            if counts[g]>=min_industry:error=max(error,abs(sums[g]/counts[g]))
        diag[t,2]=error;diag[t,3]=corr(z,x)
    return out,diag


@njit(cache=True,parallel=True)
def ema_strict(a,span):
    out=np.full(a.shape,np.nan);alpha=2./(span+1.)
    for j in prange(a.shape[1]):
        value=0.;count=0
        for t in range(a.shape[0]):
            if not np.isfinite(a[t,j]):value=0.;count=0;continue
            value=a[t,j] if count==0 else alpha*a[t,j]+(1-alpha)*value;count+=1
            if count>=span:out[t,j]=value
    return out


@njit(cache=True)
def qr_residual(y,x):
    """Pivoted, twice-reorthogonalized MGS. Handles duplicate columns exactly."""
    n,p=x.shape;a=x.copy();q=np.zeros((n,p));used=0
    for k in range(p):
        pivot=k;best=-1.
        for j in range(k,p):
            norm=0.
            for i in range(n):norm+=a[i,j]*a[i,j]
            if norm>best:best=norm;pivot=j
        if best<=n*1e-20:break
        if pivot!=k:
            temp=a[:,k].copy();a[:,k]=a[:,pivot];a[:,pivot]=temp
        for repeat in range(2):
            for j in range(used):
                dot=0.
                for i in range(n):dot+=q[i,j]*a[i,k]
                for i in range(n):a[i,k]-=dot*q[i,j]
        norm=np.sqrt((a[:,k]*a[:,k]).sum())
        if norm<=np.sqrt(n)*1e-10:continue
        q[:,used]=a[:,k]/norm
        for j in range(k+1,p):
            for repeat in range(2):
                dot=0.
                for i in range(n):dot+=q[i,used]*a[i,j]
                for i in range(n):a[i,j]-=dot*q[i,used]
        used+=1
    r=y.copy()
    for repeat in range(2):
        for j in range(used):
            dot=0.
            for i in range(n):dot+=q[i,j]*r[i]
            for i in range(n):r[i]-=dot*q[i,j]
    sd=r.std()
    if sd<=1e-10:return np.full(n,np.nan)
    return (r-r.mean())/sd


@njit(cache=True,parallel=True)
def rank_only(raw,cap,industry,eligible,min_industry,min_cross):
    """Pool validation and average ranking ONLY; no regression/zscore here."""
    out=np.full(raw.shape,np.nan);used=np.zeros(raw.shape[0])
    for t in prange(raw.shape[0]):
        valid=eligible[t]&np.isfinite(raw[t])&np.isfinite(cap[t])&(cap[t]>0)&(industry[t]>=0)
        ix=np.where(valid)[0]
        if len(ix)==0:continue
        counts=np.bincount(industry[t,ix]);ix=ix[counts[industry[t,ix]]>=min_industry];used[t]=len(ix)
        if len(ix)<min_cross:continue
        r=average_ranks(raw[t,ix])
        for k in range(len(ix)):out[t,ix[k]]=r[k]
    return out,used


@njit(cache=True,parallel=True)
def project(score,xlist,cap,industry,min_cross):
    out=np.full(score.shape,np.nan)
    for t in prange(score.shape[0]):
        ix=np.where(np.isfinite(score[t]))[0];n=len(ix)
        if n<min_cross:continue
        groups=industry[t,ix];counts=np.bincount(groups);y=score[t,ix].copy();means=np.bincount(groups,weights=y)
        for i in range(n):y[i]-=means[groups[i]]/counts[groups[i]]
        x=np.empty((n,1+len(xlist)))
        for i in range(n):
            x[i,0]=np.log(cap[t,ix[i]])
            for k in range(len(xlist)):x[i,k+1]=xlist[k][t,ix[i]]
        for k in range(x.shape[1]):
            means=np.bincount(groups,weights=x[:,k])
            for i in range(n):x[i,k]-=means[groups[i]]/counts[groups[i]]
            sd=x[:,k].std()
            if sd>1e-12:x[:,k]/=sd
            else:x[:,k]=0.
        rr=qr_residual(y,x)
        for i in range(n):out[t,ix[i]]=rr[i]
    return out


@njit(cache=True,parallel=True)
def simulate_positive(close,tradable,score,begin,last,min_cross):
    cols=close.shape[1];days=last-begin;out=np.zeros((2,days,5))
    for case in prange(2):
        direction=1.;rate=(case%3)*.001;w=np.zeros(cols);pre=np.zeros(cols);target=np.zeros(cols)
        for t in range(begin,last+1):
            gross=0.
            for j in range(cols):
                r=close[t,j]/close[t-1,j]-1 if t>0 and close[t-1,j]>0 else 0.
                if not np.isfinite(r):r=0.
                pre[j]=w[j]*(1+r);gross+=w[j]*r
            if gross<=-1:
                out[case,:,:]=np.nan
                break
            for j in range(cols):pre[j]/=1+gross
            turn=0.;fee=0.;count=0;lp=0.;sp=0.;frozen=0.
            if t>begin:
                for j in range(cols):
                    if np.isfinite(score[t-1,j]):
                        count+=1;s=direction*score[t-1,j]
                        if s>0:lp+=s
                        elif s<0:sp-=s
                    if not tradable[t,j]:frozen+=abs(pre[j])
            if t>begin and count>=min_cross and lp>0 and sp>0:
                converged=False
                for _ in range(100):
                    frozen_long=0.;frozen_short=0.
                    for j in range(cols):
                        if not tradable[t,j]:
                            target[j]=pre[j]/(1-fee)
                            if target[j]>0:frozen_long+=target[j]
                            else:frozen_short+=target[j]
                        else:target[j]=0.
                    remaining_long=max(0.,1-frozen_long);remaining_short=max(0.,1+frozen_short);turn=0.
                    for j in range(cols):
                        if tradable[t,j]:
                            s=direction*score[t-1,j]
                            if np.isfinite(s):
                                if s>0:target[j]=remaining_long*s/lp
                                elif s<0:target[j]=remaining_short*s/sp
                            turn+=abs(target[j]*(1-fee)-pre[j])
                    new_fee=rate*turn
                    if abs(new_fee-fee)<1e-14:fee=new_fee;converged=True;break
                    fee=new_fee
                if not converged:
                    out[case,:,:]=np.nan
                    break
                for j in range(cols):
                    if not tradable[t,j]:target[j]=pre[j]/(1-fee)
                    w[j]=target[j]
            else:
                for j in range(cols):w[j]=pre[j]
            if t>begin:
                row=t-begin-1;long=0.;short=0.
                for j in range(cols):
                    if w[j]>0:long+=w[j]
                    else:short+=w[j]
                out[case,row,0]=(1+gross)*(1-fee)-1;out[case,row,1]=turn;out[case,row,2]=long;out[case,row,3]=short;out[case,row,4]=frozen
    return out


@njit(cache=True,parallel=True)
def horizon_ic(close,observed,score,beg,last,horizons,min_cross):
    out=np.full((last-beg+1,len(horizons),3),np.nan)
    for off in prange(last-beg+1):
        t=beg+off
        available=0
        for j in range(close.shape[1]):
            if np.isfinite(score[t,j]):available+=1
        if available<min_cross:continue
        for ix in range(len(horizons)):
            entry=t+1;end=entry+horizons[ix]
            if end>last:continue
            x=np.empty(close.shape[1]);y=np.empty(close.shape[1]);n=0
            for j in range(close.shape[1]):
                if np.isfinite(score[t,j]) and observed[entry,j] and observed[end,j] and close[entry,j]>0 and np.isfinite(close[end,j]):
                    v=close[end,j]/close[entry,j]-1
                    if np.isfinite(v):x[n]=score[t,j];y[n]=v;n+=1
            out[off,ix,0]=n
            if n>=min_cross:out[off,ix,1]=corr(x[:n],y[:n]);out[off,ix,2]=corr(average_ranks(x[:n]),average_ranks(y[:n]))
    return out


def performance(frame,cfg):
    r=frame['net_return'].to_numpy(float)
    if not len(r): return {}
    curve=np.r_[1.,np.cumprod(1+r)]
    peaks=np.maximum.accumulate(curve); dd=1-curve/peaks
    trough=int(np.argmax(dd)); peak=int(np.argmax(curve[:trough+1]))
    dates=frame['date'].astype(str).to_numpy()
    daily_rf=(1+cfg['risk_free_annual'])**(1/cfg['annual_days'])-1
    std=np.std(r,ddof=1) if len(r)>1 else np.nan
    turnover=frame['turnover'].mean()
    return dict(from_date=dates[0],to_date=dates[-1],days=len(r),
                long_pct=100.0,short_pct=-100.0,
                annual_return_pct=(curve[-1]**(cfg['annual_days']/len(r))-1)*100,
                total_return_pct=(curve[-1]-1)*100,
                daily_turnover_pct=turnover*100,
                sharpe=np.sqrt(cfg['annual_days'])*(r.mean()-daily_rf)/std if std>1e-15 else np.nan,
                max_drawdown_pct=float(dd[trough]*100),
                dd_start=('期初' if peak==0 else dates[peak-1]) if dd[trough]>0 else '',
                dd_end=dates[trough-1] if dd[trough]>0 else '',
                daily_mean_bp=r.mean()*1e4,
                return_per_turnover_bp=r.mean()/turnover*1e4 if turnover>0 else np.nan,
                win_pct=np.mean(r>0)*100,
                avg_long_pct=frame['long_exposure'].mean()*100,
                avg_short_pct=frame['short_exposure'].mean()*100)
