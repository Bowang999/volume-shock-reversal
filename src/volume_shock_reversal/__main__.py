import argparse,json
from .research import Panel,run_research

p=argparse.ArgumentParser(description='Fixed EMA40 volume-shock reversal research. Input data are never uploaded.')
p.add_argument('--input',required=True,help='NPZ panel following data/README.md')
p.add_argument('--output',default='results/local_run')
p.add_argument('--config',default=None,help='Optional JSON containing periods/min_cross/min_industry')
a=p.parse_args();cfg=json.load(open(a.config,encoding='utf-8')) if a.config else {}
m=run_research(Panel.load(a.input),a.output,**{k:v for k,v in cfg.items() if k in ['periods','min_cross','min_industry']})
print(m[['model','period','cost_bps','annual_return_pct','sharpe','max_drawdown_pct','one_way_turnover_pct']].round(4).to_string(index=False))
