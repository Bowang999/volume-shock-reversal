"""Regenerate every empirical figure from the released aggregate CSV files."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'results/published';FIG=ROOT/'reports/figures'
PARTS={'research':('2012-03-07','2024-12-31'),'later':('2025-01-01','2026-10-20')}
COLORS={0:'#236da1',10:'#188c7a'}
NAMES={'base':'Original','pure_rev40':'Pure reversal / EMA40','control_rev40':'Control pure reversal',
       'control_attention40':'Control abnormal turnover','joint2':'Control both'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':11,'axes.titlesize':14,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#aeb9c3','text.color':'#19334b','axes.labelcolor':'#19334b','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})

def read(name):return pd.read_csv(DATA/name)

def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    for ax in fig.axes:
        conv=ax.xaxis.get_converter()
        if conv is not None and 'date' in type(conv).__name__.lower():
            loc=mdates.AutoDateLocator(minticks=3,maxticks=5);ax.xaxis.set_major_locator(loc);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.savefig(FIG/name,dpi=160,bbox_inches='tight',pad_inches=.18);plt.close(fig)

def title(fig,heading,sub):
    fig.suptitle(heading,fontsize=22,fontweight='bold',y=.985);fig.text(.5,.92,sub,ha='center',fontsize=10,color='#63899d')

def formula():
    fig=plt.figure(figsize=(12,10));ax=fig.add_axes([0,0,1,1]);ax.axis('off')
    blocks=[(.89,r'$R^{(5)}_{i,t}=\prod_{k=0}^{4}(1+r_{i,t-k})-1$',31),(.63,r'$AV_{i,t}=\frac{V_{i,t}}{\frac{1}{5}\sum_{k=1}^{5}V_{i,t-k}}-1$',33),(.37,r'$X_{i,t}=-R^{(5)}_{i,t}\max(AV_{i,t},0)$',33),(.12,r'$F_{i,t}=\operatorname{EMA}_{40}(X_{i,t})$',35)]
    for y,s,sz in blocks:ax.text(.5,y,s,ha='center',va='center',fontsize=sz)
    # The first two definitions jointly enter X; the arrows express construction.
    for a,b in [(.78,.73),(.51,.46),(.27,.20)]:ax.annotate('',xy=(.5,b),xytext=(.5,a),arrowprops=dict(arrowstyle='-|>',color='#63899d',lw=2,mutation_scale=23))
    save(fig,'01_formula.png')

def performance(gross_only=True):
    p=read('portfolio_daily.csv.gz');m=read('metrics.csv');fig=plt.figure(figsize=(15,7));gs=fig.add_gridspec(2,2,height_ratios=[4.7,1.1],left=.07,right=.985,top=.83,bottom=.06,hspace=.28,wspace=.18)
    title(fig,'Volume-Shock Reversal | '+('Gross Performance' if gross_only else 'Gross and 10 bp'),'Daily linear +100% / -100% | next-close execution | continuous holdings')
    for col,(part,(lo,hi)) in enumerate(PARTS.items()):
        ax=fig.add_subplot(gs[0,col]);rows=[]
        for cost in ([0] if gross_only else [0,10]):
            z=p[p.model.eq('base')&p.cost_bps.eq(cost)&p.date.between(lo,hi)].sort_values('date');r=m[m.model.eq('base')&m.period.eq(part)&m.cost_bps.eq(cost)].iloc[0]
            ax.plot(pd.to_datetime(z.date),np.cumprod(1+z.net_return),lw=2,color=COLORS[cost],label='Gross' if cost==0 else '10 bp')
            rows.append(['Gross' if cost==0 else '10 bp',f'{r.annual_return_pct:.2f}%',f'{r.sharpe:.2f}',f'{r.max_drawdown_pct:.2f}%',f'{r.one_way_turnover_pct:.2f}%'])
        ax.set(title=r.from_date+' to '+r.to_date,xlabel='Year',ylabel='Cumulative net value');ax.legend(frameon=False);ax.grid(alpha=.13)
        tabax=fig.add_subplot(gs[1,col]);tabax.axis('off');t=tabax.table(cellText=rows,colLabels=['Cost','CAGR','Sharpe','MDD','TO (1w)'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.6)
        for (i,j),c in t.get_celld().items():c.set_edgecolor('#dae3eb');c.set_linewidth(.5);c.set_facecolor('#edf3f7' if i==0 else 'white')
    save(fig,'02_gross_performance.png' if gross_only else '16_cost_comparison.png')

def ic_summary():
    f=read('ic_summary.csv');f=f[f.model.eq('base')];fig,axes=plt.subplots(1,2,figsize=(15,6));fig.subplots_adjust(left=.075,right=.985,top=.82,bottom=.15,wspace=.20);title(fig,'IC and Rank IC | Horizon Profile','Mean correlation with HAC 95% intervals | next-close entry | overlapping 5D / 21D labels')
    for ax,part in zip(axes,PARTS):
        for off,metric,c in [(-.18,'ic','#236da1'),(.18,'rank_ic','#188c7a')]:
            z=f[f.period.eq(part)&f.metric.eq(metric)].set_index('horizon').reindex([1,5,21]);err=np.vstack([z['mean']-z.CI95_low,z.CI95_high-z['mean']]);ax.bar(np.arange(3)+off,z['mean'],.34,color=c,label='IC (Pearson)' if metric=='ic' else 'Rank IC (Spearman)',yerr=err,capsize=3)
            for j,v in enumerate(z['mean']):ax.text(j+off,float(z.CI95_high.iloc[j])+.0025,f'{v:.3f}',ha='center',fontsize=10)
        top=f[f.period.eq(part)].CI95_high.max();bottom=f[f.period.eq(part)].CI95_low.min();ax.set(ylim=(min(0,bottom)-.004,top*1.18+.004),xticks=range(3),xticklabels=['1 session','5 sessions','21 sessions'],ylabel='Mean cross-sectional correlation',title='2012-2024' if part=='research' else '2025 to 2026-10-20');ax.axhline(0,color='gray',lw=.7);ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.13)
    save(fig,'03_ic_rankic_summary.png')

def ic_history():
    f=read('ic_daily.csv.gz');f=f[f.model.eq('base')&f.horizon.eq(1)];fig,axes=plt.subplots(2,1,figsize=(15,8));fig.subplots_adjust(left=.08,right=.985,top=.84,bottom=.09,hspace=.34);title(fig,'Daily IC and Rank IC | Time Stability','Faint line: daily Rank IC | solid lines: trailing 63-session means')
    for ax,part in zip(axes,PARTS):
        z=f[f.period.eq(part)].sort_values('date');date=pd.to_datetime(z.date);ax.plot(date,z.rank_ic,color='#b3c9d3',alpha=.30,lw=.6)
        for metric,c,label in [('ic','#236da1','IC'),('rank_ic','#188c7a','Rank IC')]:ax.plot(date,z[metric].rolling(63,min_periods=42).mean(),lw=1.8,color=c,label=label+' | 63-session mean')
        ax.axhline(0,color='gray',lw=.7);ax.set(title='2012-2024' if part=='research' else '2025 to 2026-10-20',ylabel='Cross-sectional correlation',xlabel='Year');ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.13)
    save(fig,'04_ic_time_series.png')

def cumulative_ic():
    f=read('ic_daily.csv.gz');f=f[f.model.eq('base')&f.horizon.eq(1)];fig,axes=plt.subplots(1,2,figsize=(15,6));fig.subplots_adjust(left=.075,right=.985,top=.82,bottom=.13,wspace=.20);title(fig,'Cumulative Daily IC and Rank IC','Arithmetic correlation sums are diagnostics; they are not portfolio returns or NAV')
    for ax,part in zip(axes,PARTS):
        z=f[f.period.eq(part)].sort_values('date')
        for metric,c,label in [('ic','#236da1','IC'),('rank_ic','#188c7a','Rank IC')]:ax.plot(pd.to_datetime(z.date),z[metric].fillna(0).cumsum(),color=c,lw=1.9,label=label)
        ax.set(title='2012-2024' if part=='research' else '2025 to 2026-10-20',xlabel='Year',ylabel='Cumulative correlation sum');ax.legend(frameon=False);ax.grid(alpha=.13)
    save(fig,'05_cumulative_ic.png')

def ic_distribution():
    f=read('ic_daily.csv.gz');f=f[f.model.eq('base')&f.horizon.eq(1)];fig,axes=plt.subplots(1,2,figsize=(15,6));fig.subplots_adjust(left=.075,right=.985,top=.82,bottom=.14,wspace=.20);title(fig,'Daily IC and Rank IC | Distributions','Same final factor scores and next-close entry as the portfolio study')
    for ax,part in zip(axes,PARTS):
        z=f[f.period.eq(part)]
        for metric,c,label in [('ic','#236da1','IC'),('rank_ic','#188c7a','Rank IC')]:
            a=z[metric].dropna();ax.hist(a,bins=38,density=True,alpha=.45,color=c,label=f'{label} | mean {a.mean():.4f}');ax.axvline(a.mean(),color=c,lw=1.4)
        ax.axvline(0,color='gray',ls='--',lw=.8);ax.set(title='2012-2024' if part=='research' else '2025 to 2026-10-20',xlabel='Daily correlation',ylabel='Density');ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.13)
    save(fig,'06_ic_distributions.png')

def yearly_ic():
    f=read('ic_daily.csv.gz');f=f[f.model.eq('base')&f.horizon.eq(1)].copy();f['year']=pd.to_datetime(f.date).dt.year;z=f.groupby('year')[['ic','rank_ic']].mean();fig,ax=plt.subplots(figsize=(15,6));fig.subplots_adjust(left=.075,right=.985,top=.82,bottom=.15);title(fig,'IC and Rank IC | Calendar-Year Means','2012 and 2026 are partial years | all available valid 1D labels')
    for off,metric,c,label in [(-.19,'ic','#236da1','IC'),(.19,'rank_ic','#188c7a','Rank IC')]:ax.bar(np.arange(len(z))+off,z[metric],.36,color=c,label=label)
    ax.set(xticks=range(len(z)),xticklabels=[str(y)+('*' if y in [2012,2026] else '') for y in z.index],ylabel='Mean daily correlation',xlabel='Year');ax.axhline(0,color='gray',lw=.7);ax.legend(frameon=False,ncol=2);ax.grid(axis='y',alpha=.13);save(fig,'07_yearly_ic.png')

def deciles():
    p=read('decile_daily.csv.gz');m=read('decile_metrics.csv');fig,axes=plt.subplots(1,2,figsize=(15,6));fig.subplots_adjust(left=.075,right=.985,top=.82,bottom=.14,wspace=.18);title(fig,'Gross Decile Portfolios | Independent Long-Only Accounts','G1: low score / G10: high score | equal-weight | log scale | not the main long-short book')
    for ax,part in zip(axes,PARTS):
        for g in range(1,11):
            z=p[p.period.eq(part)&p.cost_bps.eq(0)&p.group.eq(g)].sort_values('date');color='#d48735' if g==1 else '#19334b' if g==10 else plt.cm.viridis((g-1)/9);ax.plot(pd.to_datetime(z.date),np.cumprod(1+z.net_return),lw=2 if g in [1,10] else 1.1,color=color,label=f'G{g}')
        ax.set(yscale='log',title='2012-2024' if part=='research' else '2025 to 2026-10-20',xlabel='Year',ylabel='Cumulative net value (log scale)');ax.legend(frameon=False,ncol=2,fontsize=8);ax.grid(alpha=.13)
    save(fig,'08_decile_nav.png');fig,axes=plt.subplots(1,2,figsize=(15,6));fig.subplots_adjust(left=.075,right=.985,top=.82,bottom=.14,wspace=.18);title(fig,'Gross Decile Returns','Each group is an independently simulated 100%-long equal-weight portfolio')
    for ax,part in zip(axes,PARTS):
        z=m[m.period.eq(part)&m.cost_bps.eq(0)].sort_values('group');ax.bar(z.group,z.annual_return_pct,color=[plt.cm.viridis((g-1)/9) for g in z.group]);ax.set(xticks=range(1,11),xlabel='Factor-score decile: low to high',ylabel='CAGR (%)',title='2012-2024' if part=='research' else '2025 to 2026-10-20');ax.axhline(0,color='gray',lw=.7);ax.grid(axis='y',alpha=.13)
    save(fig,'09_decile_returns.png')

def parameters(cost=0):
    g=read('parameter_metrics.csv');g=g[g.cost_bps.eq(cost)&g.period.isin(PARTS)];fig,axes=plt.subplots(5,2,figsize=(12,20));fig.subplots_adjust(left=.09,right=.90,top=.92,bottom=.04,hspace=.48,wspace=.23);title(fig,'All 125 Parameters | '+('Gross Returns' if cost==0 else '10 bp Returns'),'Return 3-7 / volume reference 3-7 / EMA30-50 | centre 5 / 5 / 40 remains fixed')
    lo=min(0,g.annual_return_pct.min());hi=g.annual_return_pct.max()
    for row,e in enumerate([30,35,40,45,50]):
        for col,part in enumerate(PARTS):
            a=g[g.ema_span.eq(e)&g.period.eq(part)].pivot(index='return_window',columns='volume_window',values='annual_return_pct').reindex(index=range(3,8),columns=range(3,8)).to_numpy();ax=axes[row,col];im=ax.imshow(a,cmap='Blues',vmin=lo,vmax=hi,aspect='auto');ax.set(xticks=range(5),xticklabels=range(3,8),yticks=range(5),yticklabels=range(3,8),xlabel='Volume reference (sessions)',ylabel='Return window (sessions)',title=('2012-2024' if part=='research' else '2025 to 2026-10-20')+f' | EMA{e}')
            for i in range(5):
                for j in range(5):ax.text(j,i,f'{a[i,j]:.1f}%',ha='center',va='center',fontsize=10,color='white' if a[i,j]>hi*.65 else '#19334b')
            if e==40:ax.add_patch(Rectangle((1.5,1.5),1,1,fill=False,edgecolor='#d48735',lw=2.5))
    fig.colorbar(im,cax=fig.add_axes([.93,.07,.02,.80]),label='CAGR (%)');save(fig,'10_parameter_gross.png' if cost==0 else '17_parameter_cost10.png')

def subperiods():
    m=read('subperiod_metrics.csv');ic=read('subperiod_ic.csv');parts=['bull_2014_2015','bear_2018','structural_2019_2021','range_2022_2024'];labels=['2014-2015','2018','2019-2021','2022-2024'];fig,axes=plt.subplots(1,2,figsize=(15,6));fig.subplots_adjust(left=.075,right=.985,top=.82,bottom=.14,wspace=.20);title(fig,'Historical Subperiod Stability','Gross original strategy | fixed 5 / 5 / 40 | period names are descriptive')
    z=m[m.model.eq('base')&m.cost_bps.eq(0)].set_index('period').reindex(parts);axes[0].bar(range(4),z.annual_return_pct,color='#236da1');axes[0].set(xticks=range(4),xticklabels=labels,ylabel='Gross CAGR (%)')
    for off,metric,c in [(-.19,'ic','#236da1'),(.19,'rank_ic','#188c7a')]:
        z=ic[ic.model.eq('base')&ic.horizon.eq(1)&ic.metric.eq(metric)].set_index('period').reindex(parts);axes[1].bar(np.arange(4)+off,z['mean'],.36,label='IC' if metric=='ic' else 'Rank IC',color=c)
    axes[1].set(xticks=range(4),xticklabels=labels,ylabel='Mean daily correlation');axes[1].legend(frameon=False)
    for ax in axes:ax.axhline(0,color='gray',lw=.7);ax.grid(axis='y',alpha=.13)
    save(fig,'11_subperiod_stability.png')

def incremental():
    m=read('metrics.csv');ic=read('ic_summary.csv');names=list(NAMES);fig,axes=plt.subplots(1,2,figsize=(15,7));fig.subplots_adjust(left=.23,right=.985,top=.82,bottom=.12,wspace=.20);title(fig,'Gross Incremental-Information Diagnostics','One joint industry / float-cap OLS | only pure reversal and abnormal-turnover controls')
    for ax,part in zip(axes,PARTS):
        z=m[m.period.eq(part)&m.cost_bps.eq(0)].set_index('model').reindex(names);ax.barh(range(5),z.annual_return_pct,color=['#236da1','#8aa6ba','#188c7a','#63899d','#d48735']);ax.set(yticks=range(5),yticklabels=[NAMES[k] for k in names] if part=='research' else ['']*5,xlabel='Gross long-short CAGR (%)',title='2012-2024' if part=='research' else '2025 to 2026-10-20');ax.invert_yaxis();ax.grid(axis='x',alpha=.13)
        for j,v in enumerate(z.annual_return_pct):ax.text(v+.2,j,f'{v:.2f}%',va='center',fontsize=10)
        ax.set_xlim(0,m[m.cost_bps.eq(0)].annual_return_pct.max()*1.20)
    save(fig,'12_incremental_gross.png');fig,axes=plt.subplots(1,2,figsize=(15,7));fig.subplots_adjust(left=.23,right=.985,top=.82,bottom=.12,wspace=.20);title(fig,'Incremental Daily IC and Rank IC','Orthogonality to selected controls is constructed; predictive correlation is evaluated separately')
    for ax,part in zip(axes,PARTS):
        for off,metric,c in [(-.16,'ic','#236da1'),(.16,'rank_ic','#188c7a')]:
            z=ic[ic.period.eq(part)&ic.horizon.eq(1)&ic.metric.eq(metric)].set_index('model').reindex(names);ax.barh(np.arange(5)+off,z['mean'],.30,label='IC' if metric=='ic' else 'Rank IC',color=c)
        ax.set(yticks=range(5),yticklabels=[NAMES[k] for k in names] if part=='research' else ['']*5,xlabel='Mean daily correlation',title='2012-2024' if part=='research' else '2025 to 2026-10-20');ax.invert_yaxis();ax.legend(frameon=False);ax.grid(axis='x',alpha=.13)
    save(fig,'13_incremental_ic.png')

def smoothing():
    m=read('ema_comparison.csv');fig,axes=plt.subplots(1,2,figsize=(15,6));fig.subplots_adjust(left=.075,right=.985,top=.82,bottom=.14,wspace=.20);title(fig,'EMA Smoothing | Return and Turnover Trade-Off','EMA40 was chosen after historical comparison; the choice is not preregistered')
    for ax,metric,label in [(axes[0],'annual_return_pct','Gross CAGR (%)'),(axes[1],'one_way_turnover_pct','Daily one-way turnover (%)')]:
        for part,c in [('research','#236da1'),('later','#188c7a')]:
            z=m[m.period.eq(part)&m.cost_bps.eq(0)].sort_values('ema_span');ax.plot(z.ema_span,z[metric],marker='o',color=c,lw=1.8,label='2012-2024' if part=='research' else '2025 to 2026-10-20')
        ax.set(xticks=[20,40,60],xlabel='EMA span',ylabel=label);ax.legend(frameon=False);ax.grid(alpha=.13)
    save(fig,'14_ema_tradeoff.png')

def yearly_returns():
    p=read('portfolio_daily.csv.gz');p=p[p.model.eq('base')&p.cost_bps.eq(0)].copy();p['year']=pd.to_datetime(p.date).dt.year;z=p.groupby('year').net_return.apply(lambda x:(np.prod(1+x)-1)*100);fig,ax=plt.subplots(figsize=(15,6));fig.subplots_adjust(left=.075,right=.985,top=.82,bottom=.14);title(fig,'Gross Calendar-Year Returns','Partial 2012 / 2026 returns are not annualized | continuous holdings')
    ax.bar(range(len(z)),z,color='#236da1');ax.set(xticks=range(len(z)),xticklabels=[str(y)+('*' if y in [2012,2026] else '') for y in z.index],xlabel='Year',ylabel='Calendar-period return (%)');ax.axhline(0,color='gray',lw=.7);ax.grid(axis='y',alpha=.13);save(fig,'15_yearly_gross_returns.png')

def render_all():
    style();formula();performance();ic_summary();ic_history();cumulative_ic();ic_distribution();yearly_ic();deciles();parameters();subperiods();incremental();smoothing();yearly_returns();performance(False);parameters(10)
    print(f'Rendered {len(list(FIG.glob("*.png")))} figures from released aggregates.')

if __name__=='__main__':render_all()
