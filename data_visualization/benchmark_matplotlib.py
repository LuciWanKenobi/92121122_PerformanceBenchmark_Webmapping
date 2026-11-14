#!/usr/bin/env python3
"""Erzeuge die vierzehn Abbildungen 4 bis 17 direkt aus benchmark-combined.csv.

Aufruf: python benchmark_matplotlib.py benchmark-combined.csv --output figures

Nur abgeschlossene Durchläufe werden verwendet. warmup=true ist kein
Ausschlusskriterium. Aggregation je Framework, Pfad, Szene und Kartenanzahl.
Lokale Pickle-Dateien und die Dashboard-Anwendung werden nicht benötigt.
Abbildung 14 zeigt nur die Mediane der Karteninitialisierung (mapInitMs) im PNG-Pfad
vor dem Laden der Layer; ein Detailausschnitt trennt die nahe beieinanderliegenden Kurven.
Abbildung 15 vergleicht die Layerbereitschaft aller Frameworks in drei 3D-Diagrammen
für GRIB2, PNG und COG mit gemeinsamer Zeitskala.
Abbildung 16 zeigt Callback-Rate und Rasterabdeckung aller Frameworks im GRIB2-Pfad.
Abbildung 17 zeigt die Nachzugdauer nach Bewegungsende bis zur Raster- und
Kartenbereitschaft aller Karten (postMotionRasterCatchUpMs), ebenfalls im GRIB2-Pfad.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
KEYS = ['suiteFramework', 'suiteRasterPath', 'sceneId', 'mapCount']
METRICS = ['mapInitMs', 'layerReadyMs', 'timeChangeReadyMs', 'rafFps', 'motionRasterCoveragePercent', 'heapPeakBytes', 'systemCpuMeanPercent', 'serverRequestCount', 'serverBodyBytesSent', 'timeChangeMapReadySpreadMs', 'postMotionRasterCatchUpMs']
FRAMEWORKS = ['leaflet', 'openlayers', 'mapbox', 'maplibre']
PATHS = ['xyz', 'cog', 'grib']
SCENES = ['I', 'II_A', 'II_B', 'III', 'V', 'VIII', 'IX']
MAP_COUNTS = [1, 2, 5, 10, 16]
# Rot → Orange → Gelb → Grün = schlecht bis gut; für Zeiten wird die Skala umgekehrt.
PERFORMANCE_CMAP = LinearSegmentedColormap.from_list(
    'benchmark_bad_to_good', ['#c0392b', '#f5b041', '#ffeb3b', '#238b45']
)

def load_data(csv_path: Path, output: Path):
    """Validiere Eingabe und exportiere Mediane, Quartile und gültige Anzahlen."""
    raw = pd.read_csv(csv_path, low_memory=False)
    missing = sorted(set(KEYS + METRICS + ['recordType', 'repetition']) - set(raw.columns))
    if missing:
        raise ValueError('Erforderliche CSV-Spalten fehlen: ' + ', '.join(missing))
    completed = raw.loc[raw['recordType'].eq('completed')].copy()
    core = completed['suiteFramework'].isin(FRAMEWORKS) & completed['suiteRasterPath'].isin(PATHS) & completed['sceneId'].isin(SCENES) & completed['mapCount'].isin(MAP_COUNTS)
    data = completed.loc[core].copy()
    if data.empty:
        raise ValueError('Keine abgeschlossenen Durchläufe der Kernmatrix vorhanden.')
    if data.duplicated(KEYS + ['repetition']).any():
        raise ValueError('Mehrdeutige Wiederholungen. Zuerst gültigen Versuch pro Konfiguration auswählen.')
    for column in METRICS:
        converted = pd.to_numeric(data[column], errors='coerce')
        if (data[column].notna() & converted.isna()).any():
            raise ValueError(f'Ungültiger numerischer Wert in {column}.')
        if np.isinf(converted.dropna()).any():
            raise ValueError(f'Unendlicher Wert in {column}.')
        data[column] = converted
    grouped = data.groupby(KEYS, sort=True)
    expected = pd.MultiIndex.from_product([FRAMEWORKS, PATHS, SCENES, MAP_COUNTS], names=KEYS)
    medians = grouped[METRICS].median().reindex(expected)
    lower = grouped[METRICS].quantile(0.25, interpolation='linear').reindex(expected)
    upper = grouped[METRICS].quantile(0.75, interpolation='linear').reindex(expected)
    valid = grouped[METRICS].count().reindex(expected, fill_value=0)
    count = grouped.size().reindex(expected, fill_value=0)
    for label, values in [('median', medians), ('q1', lower), ('q3', upper), ('n', valid)]:
        values.to_csv(output / f'{label}.csv', encoding='utf-8-sig')
    count.rename('completed_repetitions').to_csv(output / 'configuration_counts.csv', encoding='utf-8-sig')
    audit = {'input_file': str(csv_path.resolve()), 'sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(), 'record_counts': raw['recordType'].value_counts(dropna=False).to_dict(), 'completed_core_records': len(data), 'completed_records_outside_core': len(completed) - len(data), 'configurations_with_five_runs': int(count.eq(5).sum()), 'configurations_with_partial_runs': int(((count > 0) & (count < 5)).sum()), 'configurations_without_runs': int(count.eq(0).sum()), 'aggregation': 'median per configuration, linearly interpolated Q1 and Q3', 'warmup_policy': 'completed rows retained regardless of warmup option', 'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__}
    (output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(data)} abgeschlossene Durchläufe der Kernmatrix ausgewertet.')
    return (data, medians, lower, upper)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path, help='Pfad zu benchmark-combined.csv')
    parser.add_argument('--output', type=Path, default=Path('figures'), help='Ausgabeordner, Standard: figures')
    parser.add_argument('--dpi', type=int, default=220, help='PNG-Auflösung, Standard: 220 dpi')
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error('--dpi muss mindestens 72 sein.')
    args.output.mkdir(parents=True, exist_ok=True)
    d, m, q1, q3 = load_data(args.csv, args.output)
    OUT = args.output
    F = ['leaflet', 'openlayers', 'mapbox', 'maplibre']
    FN = {'leaflet': 'Leaflet', 'openlayers': 'OpenLayers', 'mapbox': 'Mapbox GL JS', 'maplibre': 'MapLibre GL JS'}
    P = ['xyz', 'cog', 'grib']
    PN = {'xyz': 'PNG', 'cog': 'COG', 'grib': 'GRIB2'}
    S = ['I', 'II_A', 'II_B', 'III', 'V', 'VIII', 'IX']
    SL = ['I', 'II A', 'II B', 'III', 'V', 'VIII', 'IX']
    M = [1, 2, 5, 10, 16]
    C = {'leaflet': '#a45d00', 'openlayers': '#007b73', 'mapbox': '#3059a6', 'maplibre': '#a13a72'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.labelsize': 10, 'savefig.dpi': args.dpi, 'figure.facecolor': 'white'})
    rows = [(f, p) for f in F for p in P]

    def save(fig, name):
        options = dict(bbox_inches='tight', facecolor='white')
        labels_3d = [axis.label for ax in fig.axes if ax.name == '3d'
                     for axis in (ax.xaxis, ax.yaxis, ax.zaxis)]
        if labels_3d:
            # Matplotlib berücksichtigt projizierte 3D-Achsentitel nicht immer vollständig.
            options['bbox_extra_artists'] = fig.get_default_bbox_extra_artists() + labels_3d
        fig.savefig(OUT / (name + '.png'), **options)
        fig.savefig(OUT / (name + '.pdf'), **options)
        plt.close(fig)

    def bounds_3d(fig, axes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        return [Bbox.union([ax.get_tightbbox(renderer)] +
                           [axis.label.get_window_extent(renderer)
                            for axis in (ax.xaxis, ax.yaxis, ax.zaxis)])
                .transformed(fig.transFigure.inverted()) for ax in axes]

    def fmt(v, n=1):
        return f'{v:.{n}f}'.replace('.', ',')

    def heat(ax, metric, scale=1, cmap=PERFORMANCE_CMAP, norm=None, vmin=None, vmax=None):
        arr = np.array([[m.loc[(f, p, s, 1), metric] / scale for s in S] for f, p in rows])
        im = ax.imshow(arr, aspect='auto', cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(7), SL)
        ax.set_yticks(range(12), [FN[f] + ' · ' + PN[p] for f, p in rows], fontsize=9)
        for y in [2.5, 5.5, 8.5]:
            ax.axhline(y, color='white', lw=2)
        for i in range(12):
            for j in range(7):
                val = arr[i, j]
                if not np.isfinite(val):
                    ax.text(j, i, '–', ha='center', va='center', fontsize=8)
                    continue
                rgba = im.cmap(im.norm(val))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, fmt(val, 2 if scale == 1000 else 1), ha='center', va='center', fontsize=8, color='black' if lum > 0.55 else 'white')
        ax.set_xlabel('Szene')
        return im
    fig, ax = plt.subplots(figsize=(10.5, 6.2), layout='constrained')
    im = heat(ax, 'layerReadyMs', 1000, PERFORMANCE_CMAP.reversed(), norm=LogNorm(vmin=0.035, vmax=3))
    fig.colorbar(im, ax=ax, label='Layerbereitschaft [s]')
    save(fig, '04_initial')
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.3), layout='constrained')
    im = heat(axs[0], 'rafFps', cmap=PERFORMANCE_CMAP, vmin=0, vmax=144)
    axs[0].set_title('a  Callback-Rate [s⁻¹]')
    fig.colorbar(im, ax=axs[0], shrink=0.6)
    im = heat(axs[1], 'motionRasterCoveragePercent', cmap=PERFORMANCE_CMAP, vmin=0, vmax=100)
    axs[1].set_yticks([])
    axs[1].set_title('b  Mittlere Rasterabdeckung [%]')
    fig.colorbar(im, ax=axs[1], shrink=0.6)
    save(fig, '05_interaction')
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 6), layout='constrained')
    y = np.arange(12)
    labels = [FN[f] + ' · ' + PN[p] for f, p in rows]
    for ax, metric, scale, label in [(axs[0], 'heapPeakBytes', 2 ** 20, 'Heap-Spitzenwert [MiB]'), (axs[1], 'systemCpuMeanPercent', 1, 'Systemweite mittlere CPU-Auslastung [%]')]:
        vals = [m.loc[(f, p, 'IX', 1), metric] / scale for f, p in rows]
        lows = [q1.loc[(f, p, 'IX', 1), metric] / scale for f, p in rows]
        highs = [q3.loc[(f, p, 'IX', 1), metric] / scale for f, p in rows]
        ax.barh(y, vals, color=[C[f] for f, p in rows], xerr=[np.array(vals) - lows, np.array(highs) - vals], capsize=2)
        ax.set_yticks(y, labels if ax == axs[0] else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, '06_resources')
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 6), layout='constrained')
    for ax, metric, scale, label in [(axs[0], 'serverRequestCount', 1, 'Serverseitig gezählte Anfragen'), (axs[1], 'serverBodyBytesSent', 2 ** 20, 'Gesendete Antwortkörper [MiB]')]:
        vals = [m.loc[(f, p, 'IX', 1), metric] / scale for f, p in rows]
        ax.barh(y, vals, color=[C[f] for f, p in rows])
        ax.set_yticks(y, labels if ax == axs[0] else [], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=0.2)
        ax.set_axisbelow(True)
        for yi, v in zip(y, vals):
            ax.text(v, yi, ' ' + (str(round(v)) if scale == 1 else fmt(v, 1)), va='center', fontsize=8)
        ax.set_xlim(0, max(vals) * 1.2)
    save(fig, '07_network')
    fig, ax = plt.subplots(figsize=(10.5, 6.2), layout='constrained')
    im = heat(ax, 'timeChangeReadyMs', 1000, PERFORMANCE_CMAP.reversed(), norm=LogNorm(vmin=0.08, vmax=8))
    fig.colorbar(im, ax=ax, label='Zeitwechselbereitschaft [s]')
    save(fig, '08_update')
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.4), layout='constrained')
    for ax, p in zip(axs, P):
        for f in F:
            vals = [m.loc[(f, p, 'VIII', mc), 'layerReadyMs'] / 1000 if (f, p, 'VIII', mc) in m.index else np.nan for mc in M]
            low = [q1.loc[(f, p, 'VIII', mc), 'layerReadyMs'] / 1000 if (f, p, 'VIII', mc) in m.index else np.nan for mc in M]
            hi = [q3.loc[(f, p, 'VIII', mc), 'layerReadyMs'] / 1000 if (f, p, 'VIII', mc) in m.index else np.nan for mc in M]
            ax.plot(M, vals, 'o-', color=C[f], label=FN[f], ms=4)
            ax.fill_between(M, low, hi, color=C[f], alpha=0.15)
        ax.set_title(PN[p])
        ax.set_xticks(M)
        ax.set_xlabel('Kartenanzahl')
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 24)
    axs[0].set_ylabel('Initiale Layerbereitschaft [s]')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
    save(fig, '09_scaling')
    fig = plt.figure(figsize=(11, 8.2), layout='constrained')
    fig.get_layout_engine().set(hspace=0.18, wspace=0.10)
    axes_3d = []
    sc = ['I', 'II_B', 'III', 'V', 'VIII']
    ls = [1, 2, 3, 5, 8]
    X, Y = np.meshgrid(M, ls)
    for idx, f in enumerate(F):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        axes_3d.append(ax)
        Z = np.array([[m.loc[(f, 'grib', s, mc), 'layerReadyMs'] / 1000 for mc in M] for s in sc])
        for i in range(len(ls)):
            ax.plot(X[i], Y[i], Z[i], color=C[f], marker='o', ms=3, lw=1.1)
        for j in range(len(M)):
            ax.plot(X[:, j], Y[:, j], Z[:, j], color=C[f], lw=0.6, alpha=0.55)
        ax.set_title(FN[f], pad=1)
        ax.set_xlabel('Karten', labelpad=4)
        ax.set_ylabel('Rasterlayer', labelpad=5)
        ax.set_zlabel('Layerbereitschaft [s]', labelpad=5)
        ax.set_xticks([1, 5, 10, 16])
        ax.set_yticks([1, 3, 5, 8])
        ax.set_zlim(0, 8)
        ax.view_init(elev=25, azim=-135)
        ax.tick_params(labelsize=8, pad=0)
    # Trennlinien zwischen den vollständigen Diagrammen inklusive Titeln und Achsen.
    bounds = bounds_3d(fig, axes_3d)
    left, right = min(b.x0 for b in bounds), max(b.x1 for b in bounds)
    bottom, top = min(b.y0 for b in bounds), max(b.y1 for b in bounds)
    separator_x = (max(bounds[i].x1 for i in (0, 2))
                   + min(bounds[i].x0 for i in (1, 3))) / 2
    separator_y = (min(bounds[i].y0 for i in (0, 1))
                   + max(bounds[i].y1 for i in (2, 3))) / 2
    for xs, ys in [([separator_x, separator_x], [bottom, top]),
                   ([left, right], [separator_y, separator_y])]:
        fig.add_artist(Line2D(xs, ys, transform=fig.transFigure,
                              color='#b0b0b0', lw=0.8, in_layout=False))
    # Die gerenderte Anordnung für PNG und PDF beibehalten.
    fig.set_layout_engine('none')
    save(fig, '10_3d')
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.2), layout='constrained')
    for j, p in enumerate(P):
        for i, (metric, lab) in enumerate([('rafFps', 'Callback-Rate [s⁻¹]'), ('motionRasterCoveragePercent', 'Mittlere Rasterabdeckung [%]')]):
            ax = axs[i, j]
            for f in F:
                vals = [m.loc[(f, p, 'VIII', mc), metric] if (f, p, 'VIII', mc) in m.index else np.nan for mc in M]
                ax.plot(M, vals, 'o-', color=C[f], label=FN[f], ms=4)
            ax.set_xticks(M)
            ax.grid(alpha=0.2)
            ax.set_ylim(0, 150 if i == 0 else 105)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(PN[p])
            else:
                ax.set_xlabel('Kartenanzahl')
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
    save(fig, '11_multimotion')
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.8), layout='constrained')
    for j, p in enumerate(['xyz', 'grib']):
        for f in F:
            vals = [m.loc[(f, p, 'VIII', mc), 'timeChangeMapReadySpreadMs'] / 1000 for mc in M]
            axs[j].plot(M, vals, 'o-', color=C[f], label=FN[f], ms=4)
        axs[j].set_title(PN[p])
        axs[j].set_xticks(M)
        axs[j].set_xlabel('Kartenanzahl')
        axs[j].grid(alpha=0.2)
        axs[j].set_ylim(0, 3)
    axs[0].set_ylabel('Bereitschaftsspanne beim Zeitwechsel [s]')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
    save(fig, '12_sync')
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.5), layout='constrained')
    cmap = ListedColormap(['#b44c43', '#edba58', '#cae0ce'])
    norm = BoundaryNorm([-0.5, 0.5, 4.5, 5.5], 3)
    for ax, f in zip(axs.flat, F):
        arr = np.array([[len(d[(d.suiteFramework == f) & (d.suiteRasterPath == 'cog') & (d.sceneId == s) & (d.mapCount == mc)]) for mc in M] for s in S])
        ax.imshow(arr, aspect='auto', cmap=cmap, norm=norm)
        ax.set_xticks(range(5), M)
        ax.set_yticks(range(7), SL)
        ax.set_title(FN[f])
        ax.set_xlabel('Kartenanzahl')
        ax.set_ylabel('Szene')
        for i in range(7):
            for j in range(5):
                ax.text(j, i, str(arr[i, j]), ha='center', va='center', fontsize=11)
    save(fig, '13_completeness')
    # Native Kartenbereitschaft separat von der bereits dargestellten Layerbereitschaft.
    # Szene I hält die Konfiguration über Frameworks und Kartenanzahlen vergleichbar.
    # mapInitMs ist die gemessene Gesamtzeit der sequenziell initialisierten Karten.
    fig, ax = plt.subplots(figsize=(9, 5.6), layout='constrained')
    detail = ax.inset_axes([0.10, 0.49, 0.42, 0.42])
    styles = {'leaflet': ('--', 's', 7), 'openlayers': ('-.', '^', 5),
              'mapbox': ('-', 'o', 5), 'maplibre': (':', 'D', 4)}
    for f in F:
        vals = np.array([m.loc[(f, 'xyz', 'I', mc), 'mapInitMs'] / 1000 for mc in M])
        linestyle, marker, size = styles[f]
        # Große offene Leaflet-Symbole bleiben auch bei Überlappung mit MapLibre sichtbar.
        style = dict(color=C[f], label=FN[f], linestyle=linestyle, marker=marker,
                     ms=size, markerfacecolor='none' if f == 'leaflet' else C[f],
                     zorder=4 if f == 'leaflet' else 3)
        ax.plot(M, vals, **style)
        if f != 'mapbox':
            detail.plot(M, vals, **style)
        if f == 'mapbox' and np.isfinite(vals[-1]):
            ax.annotate(fmt(vals[-1], 2) + ' s', (M[-1], vals[-1]),
                        xytext=(-4, 9), textcoords='offset points',
                        ha='right', color=C[f], fontweight='bold')
    ax.set_xticks(M)
    ax.set_xlabel('Kartenanzahl')
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    # Lineare Hauptskala ab null zeigt den vollständigen Abstand zu Mapbox.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.set_ylabel('Initiale Kartenbereitschaft [s]')
    detail.set_title('Detail: kürzere Initialisierungszeiten', fontsize=9)
    detail.set_xticks([1, 5, 10, 16])
    detail.set_ylim(0, 0.3)
    detail.set_yticks([0, 0.1, 0.2, 0.3], ['0', '0,1', '0,2', '0,3'])
    detail.set_ylabel('Zeit [s]', fontsize=8)
    detail.tick_params(labelsize=8)
    detail.grid(alpha=0.2)
    detail.set_axisbelow(True)
    fig.suptitle('Initiale Kartenbereitschaft · PNG-Pfad · Szene I\n'
                 'Median · gesamtes Kartengitter vor dem Laden der Benchmark-Layer',
                 fontsize=12)
    fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
    save(fig, '14_map_initialization')

    # Gleiche Karten- und Layeranzahlen wie in Abbildung 10, gemeinsam je Messgröße.
    styles_3d = {'leaflet': ('--', 's'), 'openlayers': ('-.', '^'),
                 'mapbox': ('-', 'o'), 'maplibre': (':', 'D')}

    def plot_all_frameworks_3d(ax, metric, zlabel, zlim, scale=1, raster_path='grib'):
        for f in F:
            Z = np.array([[m.loc[(f, raster_path, s, mc), metric] / scale
                           for mc in M] for s in sc])
            linestyle, marker = styles_3d[f]
            for i in range(len(ls)):
                ax.plot(X[i], Y[i], Z[i], color=C[f], linestyle=linestyle,
                        marker=marker, ms=3.5, lw=1.2,
                        markerfacecolor='none' if f == 'leaflet' else C[f],
                        label=FN[f] if i == 0 else '_nolegend_')
            for j in range(len(M)):
                ax.plot(X[:, j], Y[:, j], Z[:, j], color=C[f],
                        linestyle=linestyle, lw=0.5, alpha=0.3)
        ax.set_xlabel('Karten', labelpad=5)
        ax.set_ylabel('Rasterlayer', labelpad=6)
        ax.set_zlabel(zlabel, labelpad=7)
        ax.set_xticks([1, 5, 10, 16])
        ax.set_yticks([1, 3, 5, 8])
        ax.set_zlim(*zlim)
        ax.view_init(elev=25, azim=-135)
        ax.tick_params(labelsize=8, pad=0)

    paths_3d = ['grib', 'xyz', 'cog']
    ready = m.loc[(slice(None), paths_3d, sc, slice(None)), 'layerReadyMs'] / 1000
    ready_limit = max(1, np.ceil(ready.max() * 1.1))
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.5), layout='constrained',
                             subplot_kw={'projection': '3d'})
    fig.get_layout_engine().set(wspace=0.12)
    for ax, p in zip(axs, paths_3d):
        plot_all_frameworks_3d(ax, 'layerReadyMs', 'Layerbereitschaft [s]',
                               (0, ready_limit), 1000, raster_path=p)
        ax.set_title(PN[p], pad=12)
    fig.suptitle('Layerbereitschaft · alle Frameworks · Median · gemeinsame Zeitskala',
                 fontsize=12)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center',
               ncol=4, frameon=False)
    bounds = bounds_3d(fig, axs)
    for left_panel, right_panel in zip(bounds, bounds[1:]):
        separator_x = (left_panel.x1 + right_panel.x0) / 2
        fig.add_artist(Line2D([separator_x, separator_x],
                              [min(b.y0 for b in bounds), max(b.y1 for b in bounds)],
                              transform=fig.transFigure, color='#b0b0b0', lw=0.8,
                              in_layout=False))
    fig.set_layout_engine('none')
    save(fig, '15_3d_all_frameworks')

    fig, axs = plt.subplots(1, 2, figsize=(13, 6.3), layout='constrained',
                             subplot_kw={'projection': '3d'})
    fig.get_layout_engine().set(wspace=0.12)
    for ax, metric, title, zlabel, zlim in [
            (axs[0], 'rafFps', 'Callback-Rate', 'Callback-Rate [s⁻¹]', (0, 150)),
            (axs[1], 'motionRasterCoveragePercent', 'Mittlere Rasterabdeckung',
             'Mittlere Rasterabdeckung [%]', (0, 100))]:
        plot_all_frameworks_3d(ax, metric, zlabel, zlim)
        ax.set_title(title, pad=12)
    fig.suptitle('Interaktion · alle Frameworks · GRIB2 · Median', fontsize=12)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center',
               ncol=4, frameon=False)
    bounds = bounds_3d(fig, axs)
    separator_x = (bounds[0].x1 + bounds[1].x0) / 2
    fig.add_artist(Line2D([separator_x, separator_x],
                          [min(b.y0 for b in bounds), max(b.y1 for b in bounds)],
                          transform=fig.transFigure, color='#b0b0b0', lw=0.8,
                          in_layout=False))
    fig.set_layout_engine('none')
    save(fig, '16_3d_interaction')

    # Finale Wartezeit nach Kamerastopp; umfasst Raster- und native Kartenbereitschaft.
    fig, ax = plt.subplots(figsize=(9, 7.4), layout='constrained',
                           subplot_kw={'projection': '3d'})
    catch_up = m.loc[(slice(None), 'grib', sc, slice(None)),
                     'postMotionRasterCatchUpMs'] / 1000
    catch_up_limit = max(0.1, np.ceil(catch_up.max() * 1.1 * 10) / 10)
    plot_all_frameworks_3d(ax, 'postMotionRasterCatchUpMs', 'Nachzugdauer [s]',
                           (0, catch_up_limit), 1000)
    ax.set_title('Nachzugdauer bis zur Rastervollständigkeit · GRIB2\n'
                 'Median · ab Bewegungsende bis Raster- und Kartenbereitschaft aller Karten',
                 fontsize=11, pad=12)
    fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
               ncol=4, frameon=False)
    save(fig, '17_3d_raster_catch_up')
    print('Created', len(list(OUT.glob('*.png'))), 'figures')
if __name__ == '__main__':
    main()
