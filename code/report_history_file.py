"""Write the Chinese evidence report from completed tables and independent checks."""
from pathlib import Path
import argparse,datetime,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LABELS={'forward_wor':'SeqWoR','cohort_wor':'CohortWoR','atomic_wor':'Epoch (independent)',
        'coordinated_wor':'Epoch (shared)','rebase':'Rebase','indexed_wor':'IndexWoR'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--verification',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    source=ROOT/a.analysis;out=ROOT/a.out
    summary=json.loads((source/'summary.json').read_text(encoding='utf8'))
    check=json.loads((ROOT/a.verification).read_text(encoding='utf8'));assert check['status']=='PASS'
    cells=pd.read_csv(source/'cell_summary.csv');ratios=pd.read_csv(source/'paired_ratios.csv')
    prep=pd.read_csv(source/'preparation_by_replicate.csv');life=pd.read_csv(source/'lifecycle.csv')
    budgets=pd.read_csv(source/'setup_budget_recovery.csv')
    lines=['# ICDE 2027 实验补强结果',f"\n生成日期：{datetime.date.today().isoformat()}。所有结果来自完成并核验的本地执行。",
        '\n## 执行与核验',
        f"\n共 {summary['sessions']} 个会话，其中历史选查询 {summary['primary_sessions']} 个、固定定义并发对照 {summary['controlled_sessions']} 个；共 {summary['queries']} 个查询、{summary['reports']} 次报告。",
        f"精度目标未达到的查询：{summary['censored']}；最终区间排除真值：{summary['final_exclusions']}；全部报告中排除真值：{summary['all_report_exclusions']}。所有结果均保留。",
        f"\n独立检查验证了 {check['retained_sample_pairs']:,} 个保存的采样位置/贡献对，确认与当时选定函数一致，且 WoR 不重复。{check['source_files_hash_verified']} 个物理源文件的哈希与初始记录一致；两个归一化总体与原论文使用的数据一致。",
        '\n## 一、真正随历史定义新查询的文件实验',
        '\nTPC-H / TPC-DS，初始暴露比例 0.1 / 0.9，每格 20 个配对源排列。每会话八个加权比率查询，最多四个同时活动。',
        '新查询仅在入场时，从最近 4096 条已释放记录和最近一次已提交区间中定义；定义后冻结。所有方法采用相同策略、误差预算和相对宽度目标。各自历史不同，后续目标函数可以不同，因此下表描述同一探索策略下的执行成本。',
        '\n各方法的数值为“完整续跑秒数 / FileIO API MiB”的中位数：',
        '\n| 方法 | TPC-H 10% | TPC-H 90% | TPC-DS 10% | TPC-DS 90% |',
        '|---|---:|---:|---:|---:|']
    history=cells[cells.study=='history']
    for arm,label in LABELS.items():
        values=[]
        for dataset in ('tpch','tpcds'):
            for fraction in (.1,.9):
                r=history[(history.method==arm)&(history.dataset==dataset)&(history.fraction==fraction)].iloc[0]
                values.append(f'{r.continuation_median:.3f} / {r.api_mib_median:.3f}')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['\nEpoch (shared) / SeqWoR 的完整续跑时间比，以及点位 95% 配对 bootstrap 区间：',
        '\n| 数据集 | 初始暴露 | 时间比 | 95% 区间 |','|---|---:|---:|---:|']
    shared=ratios[(ratios.study=='history')&(ratios.method=='coordinated_wor')&(ratios.metric=='continuation_seconds')]
    for r in shared.itertuples():lines.append(f'| {r.dataset} | {r.fraction:.1f} | {r.ratio:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] |')
    selected=sum(x['history_selections_verified'] for x in summary['audit'].values())
    later=sum(x['later_selections'] for x in summary['audit'].values())
    changed=sum(x['later_changed_from_initial_context'] for x in summary['audit'].values())
    lines += [f'\n已回放核验全部 {selected} 次历史选择；{later} 次后续入场中，{changed} 次函数定义不同于只使用初始上下文预定义的结果。',
        '\n置信区间以整个源排列为配对重采样单位，共 10000 次，估计的是中位数之比；不是把查询或报告当作独立重复，也不是同时置信区间。',
        '\n## 二、同信息与精度条件下的机制对照',
        '\n五个前向方法共享同一个初始缓存、源访问限制、核函数和推断实现。CohortWoR 在入场波次内共享有效样本；Rebase 对新函数计算完整前缀矩，并向全部活动查询共享后续扫描。IndexWoR 额外允许任意位置访问，明确作为更强访问对照。',
        '基线对照是本仓库的机制级实现。COSMOS、LAQy、Taster 的原系统功能与当前适配器之间的对应关系见 baseline_alignment.md；没有把适配器冒充为原系统的全量复现。',
        '\n固定定义对照使用 TPC-H、90% 初始暴露、十六个共同函数，分别允许 1/2/4/8 个活动查询。所有方法及并发窗口的查询定义、误差预算和目标均已核对一致。完整续跑中位数（秒）：',
        '\n| 活动查询数 | '+' | '.join(LABELS.values())+' |',
        '|---:|'+'---:|'*len(LABELS)]
    sweep=cells[cells.study=='concurrency']
    for w in (1,2,4,8):
        values=[f"{sweep[(sweep.active_window==w)&(sweep.method==arm)].iloc[0].continuation_median:.3f}" for arm in LABELS]
        lines.append('| '+str(w)+' | '+' | '.join(values)+' |')
    lines += ['\n## 三、准备成本、摊销模型与收益边界',
        '\n准备成本 C 包含源验证、随机排列、物化/写文件和初始前缀缓存。它是各方法共同需要支付的成本。M 表示应用对同一快照准备工作的复用次数，不能用基线数量或实验运行次数替代。',
        '\n| 数据集 | 初始暴露 | 准备成本 C 中位数（秒） | M=16 时 C/M（秒） | M=64 时 C/M（秒） |',
        '|---|---:|---:|---:|---:|']
    for (ds,fr),g in prep.groupby(['dataset','fraction']):
        c=float(g.preparation_seconds.median())
        lines.append(f'| {ds} | {fr:.1f} | {c:.3f} | {c/16:.4f} | {c/64:.4f} |')
    lines+=['\nC/M 是准备成本分摊模型。C/M + T(h) 使用指定暖启动状态的实测续跑时间，是条件成本估计，没有把源游标重置后重复运行的结果当作真实多会话轨迹。真实共享部署需要持续保留已经暴露的历史和缓存。',
        '共同准备成本不会改变两种方法续跑时间差的符号。下面的 C/Δ 是“累计续跑节省覆盖一个准备预算所需的会话数”，不是两个方法实际生命周期排序反转的阈值；Δ 非正时无有限阈值。',
        '\n| 数据集 | 初始暴露 | Shared 相对 SeqWoR 的配对节省中位数（秒） | 准备预算回收会话数 |',
        '|---|---:|---:|---:|']
    for r in budgets[budgets.method=='coordinated_wor'].itertuples():
        recovery='无有限阈值' if pd.isna(r.setup_budget_recovery_sessions) else str(int(r.setup_budget_recovery_sessions))
        lines.append(f'| {r.dataset} | {r.fraction:.1f} | {r.paired_saving_median:.4f} | {recovery} |')
    lines += ['\n归一化输入加载、JIT、完整性哈希和离线真值分别记录。FileIO API 字节描述暖 OS 缓存下的实际文件调用，不是物理设备 I/O。',
        '\n## 文件与复现',
        f"\n- 原始运行：`experiments/innovation_20261102/history_file_formal_v1/`。",
        f'- 完整分析：`{a.analysis}`。',
        f'- 独立核验：`{a.verification}`。',
        '- 绘图代码：`figures/gen_history_lifecycle.py`。',
        '- 旧实验、原图、原始数值及本轮修改前的论文均已保留。']
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(out)


if __name__=='__main__':main()
