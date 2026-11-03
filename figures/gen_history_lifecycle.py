"""Publication figure from the completed, audited history-file analysis."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',required=True)
    ap.add_argument('--out',default='figures/history_lifecycle_concurrency')
    a=ap.parse_args();source=ROOT/a.analysis;target=ROOT/a.out
    summary=json.loads((source/'summary.json').read_text(encoding='utf8'))
    assert summary['status']=='COMPLETE'
    lifecycle=pd.read_csv(source/'lifecycle.csv');ratios=pd.read_csv(source/'paired_ratios.csv')
    plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.labelsize':8.5,
        'axes.titlesize':9.5,'legend.fontsize':7.5,'axes.linewidth':.8,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'xtick.labelsize':7.5,'ytick.labelsize':7.5,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(7,2.4),facecolor='white')
    left=fig.add_axes((.08,.30,.28,.55))
    right=fig.add_axes((.45,.30,.30,.55))
    rebase_ax=fig.add_axes((.84,.30,.14,.55))
    points=[];left_handles=[]
    for dataset,color in [('tpch','#1F4E79'),('tpcds','#757C85')]:
        for fraction,marker,style in [(.1,'o','--'),(.9,'s','-')]:
            rows=lifecycle[(lifecycle.dataset==dataset)&(lifecycle.fraction==fraction)&
                           (lifecycle.method=='coordinated_wor')].sort_values('sessions')
            assert len(rows)==9
            label=('TPC-H' if dataset=='tpch' else 'TPC-DS')+f', {int(fraction*100)}%'
            line,=left.plot(rows.sessions,rows.amortized_preparation_median,color=color,
                marker=marker,ms=3.2,lw=1.1,ls=style,label=label,
                markerfacecolor='white' if fraction==.1 else color,markeredgewidth=.8)
            left_handles.append(line)
            for row in rows.itertuples():points.append(dict(panel='a',dataset=dataset,fraction=fraction,
                method='common preparation',x=row.sessions,y=row.amortized_preparation_median,
                low='',high=''))
    left.set_xscale('log',base=2);left.set_yscale('log')
    left.set_xticks([1,4,16,64,256],labels=['1','4','16','64','256'])
    left.set_xlabel('Shared-snapshot sessions, $M$',labelpad=3)
    left.set_ylabel('Preparation / session (s)',labelpad=4)
    left.set_title('(a) Common preparation',loc='left',pad=7)
    left.set_xlim(.8,320)
    fig.legend(handles=left_handles,loc='lower center',bbox_to_anchor=(.22,.003),
        ncol=2,frameon=False,handlelength=2.,columnspacing=1.,handletextpad=.5)
    methods=[('cohort_wor','CohortWoR','#757C85','D','-'),
        ('atomic_wor','Epoch indep.','#1F4E79','o','-'),
        ('coordinated_wor','Epoch shared','#3F7FA8','^','-'),
        ('rebase','Rebase','#9C755F','s','--'),
        ('indexed_wor','IndexWoR','#576B61','P',':')]
    handles=[]
    controls=ratios[(ratios.study=='concurrency') & (ratios.metric=='continuation_seconds')]
    for arm,label,color,marker,style in methods:
        ax=rebase_ax if arm=='rebase' else right
        rows=controls[controls.method==arm].sort_values('active_window')
        assert list(rows.active_window)==[1,2,4,8]
        y=rows.ratio.to_numpy();low=rows.ci_low.to_numpy();high=rows.ci_high.to_numpy()
        handle,=ax.plot(rows.active_window,y,color=color,marker=marker,ms=3.6,
            lw=1.1,ls=style,markeredgewidth=.8,label=label)
        ax.vlines(rows.active_window,low,high,color=color,lw=.8)
        ax.plot(rows.active_window,low,ls='none',marker='_',ms=3.4,color=color,mew=.8)
        ax.plot(rows.active_window,high,ls='none',marker='_',ms=3.4,color=color,mew=.8)
        if arm!='rebase':handles.append(handle)
        for row in rows.itertuples():points.append(dict(panel='c' if arm=='rebase' else 'b',dataset=row.dataset,fraction=row.fraction,
            method=arm,x=row.active_window,y=row.ratio,low=row.ci_low,high=row.ci_high))
    parity=right.axhline(1,color='#343A40',lw=.9,ls=(0,(3,2)),zorder=0,label='SeqWoR')
    for w in (1,2,4,8):points.append(dict(panel='b',dataset='tpch',fraction=.9,method='forward_wor',x=w,y=1,low=1,high=1))
    handles.append(parity)
    right.set_xticks([1,2,4,8]);right.set_xlim(.65,8.35)
    right.set_xlabel('Active queries',labelpad=3)
    right.set_ylabel('Time / SeqWoR',labelpad=4)
    right.set_title('(b) Sampling methods',loc='left',pad=7)
    samplers=controls[controls.method!='rebase']
    lower=min(.65,float(min(samplers.ci_low.min(),samplers.ratio.min()))*.94)
    upper=max(1.3,float(max(samplers.ci_high.max(),samplers.ratio.max()))*1.06)
    right.set_ylim(lower,upper)
    right.set_yticks([v for v in (.5,.75,1.,1.25,1.5,1.75,2.) if lower<=v<=upper])
    rebases=controls[controls.method=='rebase']
    rebase_ax.set_yscale('log',base=2)
    rebase_lower=min(.75,float(min(rebases.ci_low.min(),rebases.ratio.min()))*.9)
    rebase_upper=max(16.,float(max(rebases.ci_high.max(),rebases.ratio.max()))*1.15)
    rebase_ax.set_ylim(rebase_lower,rebase_upper)
    rebase_ticks=[v for v in (.125,.25,.5,1,2,4,8,16,32,64) if rebase_lower<=v<=rebase_upper]
    rebase_ax.set_yticks(rebase_ticks,labels=[f'{v:g}' for v in rebase_ticks])
    rebase_ax.set_xticks([1,2,4,8]);rebase_ax.set_xlim(.65,8.35)
    rebase_ax.set_xlabel('Active queries',labelpad=3)
    rebase_ax.set_ylabel('Time / SeqWoR (log)',labelpad=4)
    rebase_ax.set_title('(c) Rebase',loc='left',pad=7)
    rebase_ax.axhline(1,color='#343A40',lw=.9,ls=(0,(3,2)),zorder=0)
    fig.legend(handles=handles,labels=[m[1] for m in methods if m[0]!='rebase']+['SeqWoR'],
        loc='lower center',bbox_to_anchor=(.60,.003),ncol=3,frameon=False,
        handlelength=1.4,columnspacing=.8,handletextpad=.4)
    for ax in (left,right,rebase_ax):
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(length=3,width=.7,pad=2)
        ax.grid(axis='y',color='#D8DDE3',lw=.5,zorder=0)
        ax.set_axisbelow(True)
    for point in points:
        ax={'a':left,'b':right,'c':rebase_ax}[point['panel']]
        assert ax.get_xlim()[0]<=point['x']<=ax.get_xlim()[1]
        for value in (point['y'],point['low'],point['high']):
            if value!='':assert ax.get_ylim()[0]<=value<=ax.get_ylim()[1],point
    target.parent.mkdir(parents=True,exist_ok=True)
    title='Common preparation amortization and matched-query concurrency'
    fig.savefig(target.with_suffix('.pdf'),metadata={'Title':title,'Creator':'Matplotlib; AION retained experiment analysis'})
    fig.savefig(target.with_suffix('.svg'),metadata={'Title':title,'Description':'Left: C/M common preparation accounting for shared-snapshot sessions, both datasets and exposure levels. Right: all six methods with paired 95 percent intervals under identical query definitions.'})
    fig.savefig(target.with_suffix('.png'),dpi=450)
    pd.DataFrame(points).to_csv(target.with_name(target.name+'_points.csv'),index=False)
    checks=dict(analysis_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
        for name in ('summary.json','paired_ratios.csv','lifecycle.csv')},
        width_inches=7,height_inches=2.4,plotted_estimates=len(points),interval_pairs=20,
        clipping='All point estimates and interval endpoints included in axes limits.',
        outputs_sha256={ext:hashlib.sha256(target.with_suffix('.'+ext).read_bytes()).hexdigest()
                        for ext in ('pdf','svg','png')})
    target.with_name(target.name+'_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    plt.close(fig);print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
