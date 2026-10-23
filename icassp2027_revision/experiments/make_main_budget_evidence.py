#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'experiments/results_matrix/same_pilot_multibudget_summary.json').read_text())
styles={'Direct':('#d62728','o',2.5),'How2comm':('#1b9e77','s',1.3),
        'Where2comm-compatible':('#4c78a8','^',1.3),'CoSDH':('#9467bd','D',1.3)}
plt.rcParams.update({'font.family':'serif','font.size':7.4,'axes.titlesize':8.3,
                     'axes.labelsize':7.8,'legend.fontsize':6.2,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(6.8,2.45))
for name,m in d['OPV2V'].items():
    c,mk,lw=styles[name]; x=[p['payload_percent'] for p in m['points']]
    label = 'ReCAST' if name == 'Direct' else name
    axes[0].plot(x,[p['gpp'] for p in m['points']],marker=mk,color=c,lw=lw,ms=4,label=label)
axes[0].set_title('(a) OPV2V: GPP at matched payload')
axes[0].set_xlabel('Realized payload (% of method-specific Full)'); axes[0].set_ylabel('GPP')
axes[0].grid(True,alpha=.22,lw=.5); axes[0].legend(frameon=False)

m=d['V2XSet']['Direct']; x=[p['payload_percent'] for p in m['points']]
axes[1].plot(x,[100*p['ap50'] for p in m['points']],'o-',color='#d62728',lw=2.5,label='ReCAST')
axes[1].axhline(100*m['ego_ap50'],ls=':',color='#666',lw=1.1,label='Ego')
axes[1].axhline(100*m['full_ap50'],ls='--',color='#1b9e77',lw=1.1,label='Full')
axes[1].set_title('(b) V2XSet: zero-shot ReCAST budget sweep')
axes[1].set_xlabel('Realized payload (% of Full)'); axes[1].set_ylabel('AP@0.5 (%)')
axes[1].grid(True,alpha=.22,lw=.5); axes[1].legend(frameon=False)
for p in m['points']:
    axes[1].annotate(f"{100*p['ap50']:.1f}",(p['payload_percent'],100*p['ap50']),
                     xytext=(0,5),textcoords='offset points',ha='center',fontsize=6)
fig.tight_layout(w_pad=1.2)
out=ROOT/'figures'
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out/'fig_main_budget_evidence.pdf',bbox_inches='tight')
fig.savefig(out/'fig_main_budget_evidence.png',dpi=260,bbox_inches='tight')
print(out/'fig_main_budget_evidence.pdf')
