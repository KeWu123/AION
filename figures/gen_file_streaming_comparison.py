"""Draw Figure 2 from retained analysis, without experiments or bootstrap reruns.

All eight settings are shown in one aligned row order. Panels (a,b) read
existing paired ratio estimates and 95% intervals. Panel (c) divides the two
retained continuation medians; it intentionally claims no interval.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
from html import escape

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/file_streaming'
STEM = 'file_streaming_contract_resources'
ARMS = ('atomic_wor', 'coordinated_wor', 'cohort_wor')
LABELS = ('Epoch (independent)', 'Epoch (shared)', 'CohortWoR')
CELLS = [(d, fn) for d in ('tpch', 'tpcds') for fn in ('affinity', 'weighted_price')]
PALETTE = {
    'ink': '#20252B', 'grid': '#D8DDE3', 'baseline': '#8D959E',
    'atomic_wor': '#1F4E79', 'coordinated_wor': '#3F7FA8', 'cohort_wor': '#757C85',
}
MARKERS = {'atomic_wor': 'o', 'coordinated_wor': '^', 'cohort_wor': 'D'}
OFFSETS = {'atomic_wor': -.22, 'coordinated_wor': 0, 'cohort_wor': .22}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name):
    with (SRC / name).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='figures')
    args = parser.parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    cell_rows, ratio_rows = rows('cell_summary.csv'), rows('paired_ratios.csv')
    index = {(r['dataset'], r['function'], float(r['fraction']), r['method']): r for r in cell_rows}
    ratios = {(r['dataset'], r['function'], float(r['fraction']), r['metric'],
               r['numerator_arm'], r['denominator_arm']): r for r in ratio_rows}
    conditions = [(d, fn, fr) for fr in (.1, .9) for d, fn in CELLS]
    y_positions = [0, 1, 2, 3, 5, 6, 7, 8]
    labels = [('TPC-H' if d == 'tpch' else 'TPC-DS') + '\n' +
              ('Affinity' if fn == 'affinity' else 'Weighted price') for d, fn, fr in conditions]
    font = font_manager.findfont(font_manager.FontProperties(family='Arial'), fallback_to_default=False)
    style = {
        'font.family': 'Arial', 'font.size': 8.5, 'axes.labelsize': 8.5,
        'axes.titlesize': 9.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5, 'text.color': PALETTE['ink'],
        'axes.labelcolor': PALETTE['ink'], 'xtick.color': PALETTE['ink'],
        'ytick.color': PALETTE['ink'], 'axes.edgecolor': PALETTE['baseline'],
        'axes.linewidth': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': '#FFFFFF',
        'hatch.linewidth': .8,
    }
    points = []
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7, 4.45), dpi=150, facecolor='#FFFFFF')
        gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.04, .88],
                              left=.172, right=.985, bottom=.19, top=.82, wspace=.26)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        for i, ax in enumerate(axes):
            ax.set_ylim(8.7, -1.05)
            ax.set_yticks(y_positions, labels if i == 0 else [''] * 8)
            ax.tick_params(axis='y', length=0, pad=6)
            ax.tick_params(axis='x', length=3, width=.8, pad=3)
            ax.spines[['left', 'right', 'top']].set_visible(False)
            ax.grid(axis='x', color=PALETTE['grid'], linewidth=.8)
            ax.set_axisbelow(True)
            ax.axhline(4, color=PALETTE['grid'], linewidth=.8, zorder=0)
        a, b, c = axes
        a.set_title('(a) Source access', loc='left', fontweight='bold', pad=23)
        b.set_title('(b) Time vs. SeqWoR', loc='left', fontweight='bold', pad=23)
        c.set_title('(c) Time vs. Rebase', loc='left', fontweight='bold', pad=23)
        for ax in axes:
            ax.text(0, 1.035, 'Lower is better', transform=ax.transAxes, fontsize=7.5)
        a.set_xlim(0, 108)
        a.set_xticks([0, 25, 50, 75, 100], ['0', '25', '50', '75', '100'])
        a.set_xlabel('New rows / SeqWoR (%)', labelpad=6)
        b.set_xlim(.58, 1.32)
        b.set_xticks([.6, .8, 1, 1.2], ['0.6', '0.8', '1', '1.2'])
        b.set_xlabel('Continuation-time ratio', labelpad=6)
        c.set_xscale('log')
        c.set_xlim(.04, 1.9)
        c.xaxis.set_major_locator(FixedLocator([.05, .2, 1]))
        c.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:g}'))
        c.xaxis.set_minor_locator(FixedLocator([]))
        c.set_xlabel('Time ratio (log scale)', labelpad=6)
        for ax, baseline in [(a, 100), (b, 1), (c, 1)]:
            ax.axvline(baseline, color=PALETTE['baseline'], linestyle=(0, (3, 2)), linewidth=1, zorder=1)
        # Label exposure once per aligned block instead of encoding it in fill.
        for y, fraction in [(-.72, .1), (4.45, .9)]:
            a.text(-.66, y, f'h/T = {fraction:g}',
                   transform=a.get_yaxis_transform(), fontsize=8.5, fontweight='bold',
                   va='center', clip_on=False)

        for condition, center in zip(conditions, y_positions):
            d, fn, fraction = condition
            for arm in ARMS:
                row = index[(*condition, arm)]
                assert int(float(row['replicates'])) == 20
                assert int(float(row['censored_queries'])) == 0
                quality = int(float(row['all_report_exclusions'])) > 0
                y = center + OFFSETS[arm]
                color = PALETTE[arm]
                for panel, metric, ax, multiplier in [('source', 'logical_new_rows', a, 100),
                                                       ('time', 'continuation_seconds', b, 1)]:
                    r = ratios[(*condition, metric, arm, 'forward_wor')]
                    assert r['ratio_ci_defined'].lower() == 'true'
                    assert int(float(r['paired_replicates'])) == 20
                    x, lo, hi = [float(r[k]) * multiplier for k in
                                 ('ratio_of_medians', 'bootstrap_ci025', 'bootstrap_ci975')]
                    assert lo <= x <= hi and lo >= ax.get_xlim()[0] and hi <= ax.get_xlim()[1], (condition, arm, panel, lo, hi)
                    if panel == 'source':
                        ax.barh(y, x, height=.18, color=color, alpha=.88, edgecolor=color,
                                linewidth=.8, hatch='////' if arm == 'coordinated_wor' else None, zorder=2)
                        ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], color=PALETTE['ink'],
                                    linewidth=.8, capsize=1.3, capthick=.8, zorder=3)
                    else:
                        ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt=MARKERS[arm],
                                    markersize=3.8, color=color, markeredgewidth=.8,
                                    markerfacecolor='#FFFFFF' if arm != 'atomic_wor' else color,
                                    linewidth=1, capsize=1.5, capthick=.8, zorder=3)
                    if quality:
                        ax.annotate('\u2021', (hi, y), xytext=(2, 1), textcoords='offset points',
                                    fontsize=7.5, color=PALETTE['ink'], va='center', zorder=4)
                    points.append(dict(panel=panel, dataset=d, function=fn, fraction=fraction,
                                       numerator_arm=arm, denominator_arm='forward_wor',
                                       source_metric=metric, ratio=float(r['ratio_of_medians']),
                                       ci025=float(r['bootstrap_ci025']), ci975=float(r['bootstrap_ci975']),
                                       numerator_value=float(r['numerator_median']),
                                       denominator_value=float(r['denominator_median']),
                                       plotted_x=x, source_file='paired_ratios.csv',
                                       retained_truth_exclusion=quality, paired_replicates=20))
            q = index[(*condition, 'atomic_wor')]
            rb = index[(*condition, 'rebase')]
            value = float(q['continuation_seconds_median']) / float(rb['continuation_seconds_median'])
            quality = int(float(q['all_report_exclusions'])) > 0 or int(float(rb['all_report_exclusions'])) > 0
            c.plot(value, center, marker='o', markersize=4.1, color=PALETTE['atomic_wor'], zorder=3)
            label = f'{value:.2f}' + ('\u2021' if quality else '')
            # The rightmost early weighted-price case leaves room for a direct label.
            c.annotate(label, (value, center), xytext=(5, 0), textcoords='offset points',
                       va='center', ha='left', fontsize=7.5, color=PALETTE['ink'],
                       bbox=dict(facecolor='#FFFFFF', edgecolor='none', pad=.2))
            points.append(dict(panel='prefix', dataset=d, function=fn, fraction=fraction,
                               numerator_arm='atomic_wor', denominator_arm='rebase',
                               source_metric='continuation_seconds_median', ratio=value,
                               numerator_value=float(q['continuation_seconds_median']),
                               denominator_value=float(rb['continuation_seconds_median']),
                               ci025='', ci975='', plotted_x=value, source_file='cell_summary.csv',
                               retained_truth_exclusion=quality, paired_replicates=20))

        handles = [Line2D([0], [0], marker=MARKERS[arm], color=PALETTE[arm],
                          markerfacecolor='#FFFFFF' if arm != 'atomic_wor' else PALETTE[arm],
                          markeredgewidth=.8, markersize=4, linewidth=1, label=name)
                   for arm, name in zip(ARMS, LABELS)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.565, .995),
                   ncol=3, frameon=False, columnspacing=1.5, handletextpad=.45, handlelength=1.4)
        fig.text(.172, .048, '(a,b) Pointwise 95% paired-bootstrap intervals; (c) point estimates only.', fontsize=7.5)
        fig.text(.172, .014, '\u2021 Retained report excludes truth. All 20 replicates remain included.', fontsize=7.5)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        clipped = []
        tick_live = {}
        for ax in axes:
            for axis, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
                for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                    for lab in [tick.label1, tick.label2]:
                        tick_live[id(lab)] = min(lim) <= tick.get_loc() <= max(lim)
        sizes = set()
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_visible() or not text.get_text() or tick_live.get(id(text)) is False:
                continue
            sizes.add(text.get_fontsize())
            box = text.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width+.5 or box.y1 > fig.bbox.height+.5:
                clipped.append(text.get_text())
        assert not clipped, clipped
        assert min(sizes) >= 7.5 and len(sizes) <= 3, sizes
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(out / (STEM + '.' + ext), dpi=450, facecolor='#FFFFFF')
        svg = out / (STEM + '.svg')
        svg_text = svg.read_text(encoding='utf8')
        title = 'Source access and continuation time for eight matched streaming settings'
        description = ('The top and bottom row blocks have initial exposure 0.1 and 0.9. '
                       'Panel a shows source-row consumption relative to SeqWoR, for three samplers. '
                       'Panel b shows their continuation-time ratios to SeqWoR. '
                       'Panel c shows Epoch independent divided by Rebase continuation time on a log scale. '
                       'Lower values mean less work or time. Panels a and b include retained pointwise 95 percent '
                       'paired-bootstrap intervals from 20 source permutations; panel c has point estimates only. '
                       'A double dagger marks a retained report excluding truth.')
        svg_text = svg_text.replace('<svg ', '<svg role="img" aria-labelledby="fig2-title fig2-desc" ', 1)
        svg_text = svg_text.replace('<defs>', '<title id="fig2-title">' + escape(title) + '</title>\n'
                                    '<desc id="fig2-desc">' + escape(description) + '</desc>\n<defs>', 1)
        svg.write_text(svg_text, encoding='utf8')
        # A grayscale rendering is a review aid, not an accessibility certification.
        from PIL import Image
        Image.open(out / (STEM + '.png')).convert('L').save(out / (STEM + '_grayscale.png'))
        plt.close(fig)

    keys = list(dict.fromkeys(k for p in points for k in p))
    with (out / (STEM + '_points.csv')).open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(points)
    provenance = {
        'size_inches': [7, 4.45], 'width_mm': 177.8, 'minimum_font_pt': 7.5,
        'font': font, 'text_sizes': sorted(sizes), 'plotted_points': len(points),
        'conditions': 8, 'source_points': 24, 'time_points': 24, 'prefix_points': 8,
        'replicate_unit': 'source permutation', 'paired_replicates': 20,
        'interval_source': 'existing paired_ratios.csv; no bootstrap recomputation',
        'panel_a_transform': '100 * median(method rows) / median(SeqWoR rows)',
        'panel_b_transform': 'median(method time) / median(SeqWoR time)',
        'panel_c_transform': 'median(Epoch independent time) / median(Rebase time); no CI claimed',
        'input_sha256': {name: sha(SRC / name) for name in ('cell_summary.csv', 'paired_ratios.csv')},
        'script_sha256': sha(Path(__file__)), 'matplotlib_version': matplotlib.__version__,
        'figure_contract_sha256': sha(ROOT / 'figures/fig2-contract.yaml'),
        'palette': PALETTE, 'clipped_text': clipped, 'all_censors_retained': True,
        'quality_marks': 'double dagger: retained report excludes truth',
        'export_sha256': {ext: sha(out / (STEM + '.' + ext)) for ext in ('pdf', 'svg', 'png')},
    }
    (out / (STEM + '_checks.json')).write_text(json.dumps(provenance, indent=2), encoding='utf8')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
