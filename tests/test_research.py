"""Meaningful timing, accounting, masking and regression checks."""
from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volume_shock_reversal import Panel,build_signals
from volume_shock_reversal.research import raw_signals
from volume_shock_reversal.engine import ema_strict,simulate_positive,horizon_ic,preprocess_panel

def fixture(days=180,stocks=80):
    rng=np.random.default_rng(1640);r=rng.normal(0,.015,(days,stocks));close=10*np.exp(np.cumsum(r,axis=0));shares=np.broadcast_to(rng.lognormal(18,.4,stocks),(days,stocks)).copy();volume=rng.lognormal(13,.5,(days,stocks));industry=np.broadcast_to(np.arange(stocks)%5,(days,stocks)).copy();observed=np.ones(close.shape,bool)
    return Panel(pd.bdate_range('2023-01-02',periods=days).to_numpy(dtype='datetime64[D]'),close,volume,shares,close*shares,industry,observed,observed.copy())

def reference_rebalance(pre,score,tradable,min_cross,cost):
    valid=np.isfinite(score)
    if valid.sum()<min_cross:return pre.copy(),0.,0.
    longs=np.flatnonzero(valid&(score>0));shorts=np.flatnonzero(valid&(score<0))
    if not len(longs) or not len(shorts):return pre.copy(),0.,0.
    long_shape=score[longs]/score[longs].sum();short_shape=-score[shorts]/(-score[shorts]).sum();fee=0.;frozen=~tradable
    for _ in range(100):
        target=np.zeros(len(pre));target[frozen]=pre[frozen]/(1-fee);L=max(0,1-target[target>0].sum());S=max(0,1+target[target<0].sum());target[longs[tradable[longs]]]=L*long_shape[tradable[longs]];target[shorts[tradable[shorts]]]=-S*short_shape[tradable[shorts]];orders=target*(1-fee)-pre;orders[frozen]=0;turn=np.abs(orders).sum();new=cost/10000*turn
        if abs(new-fee)<1e-14:fee=new;break
        fee=new
    else:raise RuntimeError('Cost fixed point failed')
    target[frozen]=pre[frozen]/(1-fee);return target,turn,fee

class ResearchTests(unittest.TestCase):
    def test_formula_prefix_and_scale(self):
        p=fixture();raw=raw_signals(p);short=Panel(**{k:(v[:120] if isinstance(v,np.ndarray) else v) for k,v in vars(p).items()});prefix=raw_signals(short)
        for k in raw:np.testing.assert_allclose(prefix[k],raw[k][:120],atol=2e-12,equal_nan=True)
        scaled=Panel(**vars(p));scaled.volume=p.volume*31
        for k,v in raw_signals(scaled).items():np.testing.assert_allclose(v,raw[k],atol=3e-12,equal_nan=True)

    def test_literal_volume_reference_excludes_today(self):
        p=fixture(days=120,stocks=5);r=np.full(p.close.shape,np.nan);r[1:]=p.close[1:]/p.close[:-1]-1;x=np.full(p.close.shape,np.nan)
        for t in range(5,len(r)):x[t]=-(np.prod(1+r[t-4:t+1],axis=0)-1)*np.maximum(p.volume[t]/p.volume[t-5:t].mean(0)-1,0)
        np.testing.assert_allclose(raw_signals(p)['factor'],ema_strict(x,40),atol=3e-12,equal_nan=True)

    def test_ema_warmup_and_reset(self):
        x=np.ones((110,2));x[51,0]=np.nan;y=ema_strict(x,40);self.assertTrue(np.isnan(y[:39]).all());self.assertTrue(np.isfinite(y[39:51]).all());self.assertTrue(np.isnan(y[51:91,0]).all());self.assertEqual(y[91,0],1.)

    def test_joint_full_dummy_ols(self):
        p=fixture();scores,pool=build_signals(p,min_cross=50,min_industry=3);raw=raw_signals(p)
        for t in [100,150]:
            ix=np.flatnonzero(pool[t]);target=pd.Series(raw['factor'][t,ix]).rank().to_numpy();B=pd.Series(raw['reversal'][t,ix]).rank().to_numpy();H=pd.Series(raw['attention'][t,ix]).rank().to_numpy();g=p.industry[t,ix];X=np.c_[np.eye(5)[g],np.log(p.float_cap[t,ix]),B,H];sd=X.std(0);X=np.divide(X,sd,out=X.copy(),where=sd>1e-12);res=target-X@np.linalg.lstsq(X,target,rcond=1e-10)[0];res=(res-res.mean())/res.std();np.testing.assert_allclose(res,scores['joint2'][t,ix],atol=2e-8,rtol=2e-8)

    def test_next_close_timing(self):
        close=np.full((6,4),10.);close[1:,0]=20.;trade=np.ones(close.shape,bool);score=np.tile([1.,1.,-1.,-1.],(6,1));a=simulate_positive(close,trade,score,0,5,4)
        self.assertEqual(a[0,0,0],0.)  # no capture of the jump before first fill
        close[2:,0]=40.;b=simulate_positive(close,trade,score,0,5,4);self.assertAlmostEqual(b[0,1,0],.5)

    def test_failed_budget_is_not_redistributed(self):
        close=np.ones((5,4))*10;trade=np.ones(close.shape,bool);trade[1,0]=False;score=np.tile([1.,1.,-1.,-1.],(5,1));a=simulate_positive(close,trade,score,0,4,4)
        self.assertAlmostEqual(a[0,0,2],.5);self.assertAlmostEqual(a[0,0,3],-1.)

    def test_full_ledger_with_freezes_and_fees(self):
        rng=np.random.default_rng(1705);close=10*np.exp(np.cumsum(rng.normal(0,.014,(125,15)),axis=0));trade=rng.uniform(size=close.shape)>.14;score=rng.normal(size=close.shape);score[30:33]=np.nan;score[65,:5]=np.nan;a=simulate_positive(close,trade,score,2,120,8)
        for case,cost in enumerate([0,10]):
            w=np.zeros(15);rows=[]
            for t in range(3,121):
                r=close[t]/close[t-1]-1;gross=w@r;pre=w*(1+r)/(1+gross);frozen=np.abs(pre[~trade[t]]).sum();w,turn,fee=reference_rebalance(pre,score[t-1],trade[t],8,cost);rows.append([(1+gross)*(1-fee)-1,turn,w[w>0].sum(),w[w<0].sum(),frozen])
            np.testing.assert_allclose(rows,a[case],atol=5e-11,rtol=5e-10)

    def test_ic_endpoints_and_spearman_ties(self):
        close=np.array([[10,10,10,10],[20,10,10,10],[40,8,10,11],[40,8,10,11]],float);score=np.tile([1.,1.,2.,3.],(4,1));observed=np.ones(close.shape,bool);a=horizon_ic(close,observed,score,0,3,np.array([1,5]),3);ret=close[2]/close[1]-1;ref=np.corrcoef(pd.Series(score[0]).rank(),pd.Series(ret).rank())[0,1];self.assertAlmostEqual(a[0,0,2],ref);self.assertTrue(np.isnan(a[:,1]).all());self.assertTrue(np.isnan(a[-2:,0]).all())

    def test_prelisting_nan_is_allowed_but_no_later_missing_marks(self):
        p=fixture();p.close[:5,0]=np.nan;p.observed[:5,0]=False;p.tradable[:5,0]=False;p.validate();p.close[10,0]=np.nan;p.observed[10,0]=False;p.tradable[10,0]=False
        with self.assertRaises(ValueError):p.validate()

if __name__=='__main__':unittest.main()
