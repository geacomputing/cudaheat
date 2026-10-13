#!/usr/bin/env python3
"""CUDAHeat | Refined plots: 16:9, 400-DPI presentation figures.

Run: python refined_plots.py
Reads measured JSON tables from TABLES_DIR. Creates 'Refined plots' next to this
script. Add plotting functions to PLOT_REGISTRY / ENABLED_PLOTS for future plots.
Requires matplotlib and numpy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter

TABLES_DIR = Path('/home/user/Documents/Gea/cudaheat/benchmark_results/full_pipeline_presentation_20261012T144152Z/tables')
OUTPUT_DIR = Path(__file__).resolve().parent / 'Refined plots'
DPI = 400
FIG_W, FIG_H = 13.32, 7.4925  # 5328 x 2997 px; exactly 16:9
BG, WHITE, INK, MUTED = '#F7F9FB', '#FFFFFF', '#17283B', '#617287'
BLUE, TEAL, ORANGE, PURPLE = '#2475B6', '#087F8C', '#E36B45', '#7567A4'
NAVY = "#17283B"
BORDER, LIGHT = '#DDE6ED', '#B3C2D0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.unicode_minus':False})


IO_COLOR       = "#E36B45"  # Orange — Host NetCDF I/O
H2D_COLOR      = "#7567A4"  # Purple — CPU → GPU
COMPUTE_COLOR  = "#2475B6"  # Blue — GPU computation
D2H_COLOR      = "#087F8C"  # Teal — GPU → CPU



def records(filename):
    path = TABLES_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing benchmark file: {path}\nEdit TABLES_DIR at the top of refined_plots.py.')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,list) or not data:
        raise ValueError(f'Expected non-empty JSON array in {path}')
    return data


def frame(title, subtitle, number, kicker='CUDAHeat  /  BENCHMARK RESULTS'):
    fig = plt.figure(figsize=(FIG_W,FIG_H),dpi=DPI,facecolor=BG)
    fig.text(.067,.915,kicker,fontsize=11.2,color=TEAL,weight='bold',va='center')
    fig.text(.067,.834,title,fontsize=28,color=INK,weight='bold',va='center')
    fig.text(.068,.775,subtitle,fontsize=12.2,color=MUTED,va='center')
    panel=FancyBboxPatch((.056,.163),.888,.548,boxstyle='round,pad=0.007,rounding_size=0.016',
                         transform=fig.transFigure,facecolor=WHITE,edgecolor=BORDER,lw=.85,zorder=0)
    fig.add_artist(panel)
    fig.text(.932,.053,f'{number:02d}  /  CUDAHeat',color=MUTED,fontsize=9.5,ha='right')
    return fig


def axis(fig, bounds=(.125,.267,.735,.35), logx=False):
    ax=fig.add_axes(bounds,facecolor=WHITE,zorder=2)
    if logx:ax.set_xscale('log')
    ax.set_axisbelow(True)
    ax.grid(axis='y',color=BORDER,lw=.8,alpha=.85)
    ax.tick_params(axis='both',length=0,pad=10,labelcolor=MUTED,labelsize=10)
    ax.minorticks_off()
    for s in ax.spines.values():s.set_visible(False)
    return ax


def caption(fig, takeaway, source):
    fig.text(.068,.105,'THE TAKEAWAY',color=TEAL,fontsize=10.2,weight='bold')
    fig.text(.195,.105,takeaway,color=INK,fontsize=11.2)
    fig.text(.068,.053,f'Source: {source}  ·  measured benchmark results',color=MUTED,fontsize=9.2)


def save(fig,filename):
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    path=OUTPUT_DIR/filename
    fig.savefig(path,dpi=DPI,bbox_inches=None,pad_inches=0)
    plt.close(fig)
    print('Saved:',path)


def plot_gpu_acceleration():
    d=sorted(records('throughput_summary.json'),key=lambda r:r['npoints'])
    x=np.array([r['npoints'] for r in d]); compute=np.array([r['compute_speedup'] for r in d]); pipeline=np.array([r['pipeline_speedup'] for r in d])
    fig=frame('GPU acceleration, in perspective','Compute-only acceleration versus complete-pipeline speedup',1)
    ax=axis(fig,logx=True)
    ax.set_xlim(x[0]*.72,x[-1]*1.8);ax.set_ylim(0,51)
    ax.set_xticks(x);ax.set_xticklabels([f'{i:,}' for i in x]);ax.set_yticks(range(0,51,10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.0f}×'))
    ax.fill_between(x,pipeline,compute,color=BLUE,alpha=.075)
    for y,c in ((compute,BLUE),(pipeline,ORANGE)):
        ax.plot(x,y,lw=3,color=c,marker='o',ms=7,mfc='white',mew=2.1,zorder=4)
    ax.annotate(f'{compute[-1]:.1f}×  COMPUTE',(x[-1],compute[-1]),xytext=(13,0),textcoords='offset points',color=BLUE,weight='bold',fontsize=10.5,va='center')
    ax.annotate(f'{pipeline[-1]:.1f}×  PIPELINE',(x[-1],pipeline[-1]),xytext=(13,0),textcoords='offset points',color=ORANGE,weight='bold',fontsize=10.5,va='center')
    ax.set_xlabel('Spatial points processed  ·  68,016 hourly values per point',color=MUTED,labelpad=16)
    fig.text(.075,.672,'SPEEDUP RELATIVE TO CPU',color=MUTED,fontsize=10,weight='bold')
    caption(fig,'Fast GPU kernels do not guarantee a fast pipeline: host I/O remains decisive.','throughput_summary.json')
    save(fig,'01_gpu_acceleration.png')


def plot_batch_shape():
    d=sorted((r for r in records('performance_summary.json') if r['scenario']=='performance_scaling'),key=lambda r:r['npoints'])
    x=np.array([r['npoints'] for r in d]); b=np.array([r['actual_batch_size_mean'] for r in d]); batches=np.ceil(x/b).astype(int)
    fig=frame('Adaptive spatial batching','Full temporal extent is preserved as the spatial workload increases',2)
    ax=axis(fig,bounds=(.128,.270,.73,.340),logx=True)
    ax.set_xlim(x[0]*.70,x[-1]*1.55);ax.set_ylim(0,max(b)*1.36)
    ax.set_xticks(x);ax.set_xticklabels([f'{v:,}' for v in x]);ax.set_yticks([0,500,1000,1500,2000,2500]);ax.set_ylabel('Points per GPU batch',color=MUTED,labelpad=13)
    ax.plot(x,b,color=BLUE,lw=3,marker='o',ms=8,mfc='white',mew=2.2,zorder=5)
    ax.fill_between(x,0,b,color=BLUE,alpha=.07)
    # Separate, staggered annotations prevent collisions and communicate batch count.
    for i,(xx,bb,n) in enumerate(zip(x,b,batches)):
        dy=15 if i%2==0 else 28
        ax.annotate(f'{int(bb):,} pts', (xx,bb),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=9.6,color=INK,weight='bold')
        ax.annotate(f'{n} batch'+('es' if n!=1 else ''),(xx,bb),xytext=(0,-23),textcoords='offset points',ha='center',fontsize=9.1,color=MUTED)
    ax.set_xlabel('Total spatial points in the workload',color=MUTED,labelpad=16)
    fig.text(.075,.671,'AUTOMATIC GPU BATCH SIZE',color=MUTED,fontsize=10,weight='bold')
    fig.text(.752,.671,'TIME AXIS: 68,016 h',color=TEAL,fontsize=9.8,weight='bold',ha='right')
    caption(fig,'Batch width adapts to VRAM; every batch retains the entire hourly time axis.','performance_summary.json')
    save(fig,'02_adaptive_batch_shape.png')


def plot_pipeline_components():
    d=sorted(records('pipeline_breakdown.json'),key=lambda r:r['npoints'])
    x=np.arange(len(d));labels=[f"{r['npoints']:,}" for r in d]
    keys=[('host_read_seconds','Host I/O',ORANGE),('transfer_to_device_seconds','CPU → GPU',PURPLE),('gpu_compute_seconds','GPU compute',BLUE),('transfer_from_device_seconds','GPU → CPU',TEAL)]
    fig=frame('Anatomy of the processing pipeline','Measured GPU pipeline stages across increasing spatial workloads',3)
    ax=axis(fig,bounds=(.13,.275,.72,.335))
    cpu=np.array([r['npoints'] for r in d],dtype=float)
    # CPU-only compute baseline is plotted beside, never stacked onto, the GPU pipeline.
    perf={r['npoints']:r for r in records('performance_summary.json') if r['scenario']=='performance_scaling'}
    cpu=np.array([perf[r['npoints']]['cpu_compute_seconds_mean'] for r in d])
    ax.bar(x-.20,cpu,width=.34,color=LIGHT,label='CPU compute (reference)',zorder=3)
    bottom=np.zeros(len(d))
    for k,label,color in keys:
        v=np.array([r[k] for r in d]);ax.bar(x+.20,v,bottom=bottom,width=.34,color=color,label=label,zorder=3);bottom+=v
    ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_ylabel('Execution time (s)',color=MUTED,labelpad=12)
    ax.set_ylim(0,max(max(bottom),max(cpu))*1.22)
    ax.set_xlabel('Spatial points processed',color=MUTED,labelpad=14)
    ax.legend(handles=[Patch(color=LIGHT,label='CPU compute')]+[Patch(color=c,label=l) for _,l,c in keys],loc='upper left',bbox_to_anchor=(0,1.19),ncol=5,frameon=False,fontsize=9.1,columnspacing=1.1)
    # CPU compute is an alternative reference, NOT an additive GPU pipeline stage.
    fig.text(.075,.670,'GPU PIPELINE BREAKDOWN',color=MUTED,fontsize=10,weight='bold')
    fig.text(.75,.67,'LEFT: CPU  /  RIGHT: GPU pipeline',color=MUTED,fontsize=9.4,ha='right')
    caption(fig,'Host I/O dominates GPU runtime; CPU compute is shown separately as a reference.','pipeline_breakdown.json + performance_summary.json')
    save(fig,'03_pipeline_breakdown.png')


def plot_precision_comparison():
    d={r['dtype']:r for r in records('performance_summary.json') if r['scenario']=='precision_performance'}
    if not {'float32','float64'}<=d.keys():raise ValueError('Precision performance records missing')
    validation=records('numerical_consistency.json')
    threshold={r['gpu_dtype']:r for r in validation if r['record_type']=='float_field' and r['field']=='threshold'}
    fig=frame('Precision versus performance','Float32 and float64 GPU execution, validated against the CPU reference',4)
    ax=axis(fig,bounds=(.133,.270,.38,.34))
    x=np.arange(2);time=[d[p]['pipeline_wall_seconds_mean'] for p in ('float32','float64')]
    bars=ax.bar(x,time,width=.47,color=[BLUE,PURPLE],zorder=3)
    ax.set_xticks(x);ax.set_xticklabels(['float32','float64'],fontsize=11,color=INK)
    ax.set_ylim(0,max(time)*1.38);ax.set_ylabel('Pipeline time (s)',color=MUTED,labelpad=12)
    for bar,val in zip(bars,time):ax.text(bar.get_x()+bar.get_width()/2,val+1,f'{val:.1f} s',ha='center',color=INK,weight='bold',fontsize=12)
    ax2=axis(fig,bounds=(.608,.270,.255,.34))
    errors=[threshold[p]['max_absolute_error'] for p in ('float32','float64')]
    ax2.bar(x,errors,width=.47,color=[BLUE,PURPLE],zorder=3)
    ax2.set_yscale('log');ax2.set_ylim(1e-12,1e-3)
    ax2.set_xticks(x);ax2.set_xticklabels(['float32','float64'],fontsize=11,color=INK)
    ax2.set_ylabel('Maximum |threshold error| (K)',color=MUTED,labelpad=12,fontsize=9.8)
    for xx,v in zip(x,errors):ax2.text(xx,v*2.0,f'{v:.1e}',ha='center',fontsize=10,color=INK,weight='bold')
    fig.text(.139,.669,'FULL PIPELINE',color=MUTED,fontsize=10,weight='bold')
    fig.text(.615,.669,'THRESHOLD NUMERICAL ERROR',color=MUTED,fontsize=10,weight='bold')
    caption(fig,'Float32 reduces runtime; precision effects are quantified against CPU float64.','performance_summary.json + numerical_consistency.json')
    save(fig,'04_precision_comparison.png')


def plot_pipeline_speedup():
    d=sorted((r for r in records('performance_summary.json') if r['scenario']=='performance_scaling'),key=lambda r:r['npoints'])
    x=np.array([r['npoints'] for r in d]);y=np.array([r['pipeline_speedup_mean'] for r in d])
    fig=frame('Real-world acceleration','Complete-pipeline speedup relative to CPU, including host I/O',5)
    ax=axis(fig,logx=True)
    ax.set_xlim(x[0]*.70,x[-1]*1.6);ax.set_ylim(0,max(y)*1.38)
    ax.set_xticks(x);ax.set_xticklabels([f'{v:,}' for v in x]);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}×'))
    ax.plot(x,y,color=ORANGE,lw=3.2,marker='o',ms=8,mfc='white',mew=2.3,zorder=4)
    ax.fill_between(x,0,y,color=ORANGE,alpha=.065)
    for i,(xx,yy) in enumerate(zip(x,y)):
        ax.annotate(f'{yy:.1f}×',(xx,yy),xytext=(0,16 if i%2==0 else -23),textcoords='offset points',ha='center',fontsize=11,color=INK,weight='bold')
    ax.set_xlabel('Spatial points processed  ·  68,016 hourly values per point',color=MUTED,labelpad=16)
    ax.set_ylabel('Complete-pipeline speedup',color=MUTED,labelpad=13)
    fig.text(.075,.672,'CPU BASELINE  =  1×',color=MUTED,fontsize=10,weight='bold')
    caption(fig,'Pipeline acceleration is substantial, but remains below compute-only speedup.','performance_summary.json')
    save(fig,'05_pipeline_speedup.png')



def plot_workload_scaling():
    """Measured execution time scaling with spatial workload."""

    d = sorted(
        (
            r for r in records("performance_summary.json")
            if r.get("scenario") == "performance_scaling"
        ),
        key=lambda r: r["npoints"]
    )

    x = np.array([r["npoints"] for r in d], dtype=float)
    cpu = np.array([r["cpu_compute_seconds_mean"] for r in d])
    gpu = np.array([r["gpu_compute_seconds_mean"] for r in d])
    wall = np.array([r["pipeline_wall_seconds_mean"] for r in d])

    fig = frame(
        "Scaling with spatial workload",
        "Execution time across 32–22,083 points  ·  "
        "fixed 68,016-hour record per point",
        6
    )

    ax = axis(fig, bounds=(.14, .275, .69, .335), logx=True)

    ax.set_yscale("log")
    ax.set_xlim(x[0] * .72, x[-1] * 2.05)
    ax.set_ylim(min(gpu) * .46, max(cpu) * 2.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v):,}" for v in x])

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: f"{v:g}" if v >= .1 else f"{v:.2g}"
        )
    )

    ax.grid(
        axis="y",
        which="major",
        color=BORDER,
        lw=.8,
        alpha=.85
    )

    for values, color in (
        (cpu, PURPLE),
        (wall, ORANGE),
        (gpu, BLUE)
    ):
        ax.plot(
            x, values,
            lw=3.1,
            color=color,
            marker="o",
            ms=7.5,
            mfc="white",
            mew=2.1,
            zorder=5
        )

    # Direct labels placed beyond the last data point.
    for values, color, label in (
        (cpu, PURPLE, "CPU compute"),
        (wall, ORANGE, "GPU pipeline"),
        (gpu, BLUE, "GPU compute")
    ):
        ax.annotate(
            f"{values[-1]:.1f} s  {label}",
            (x[-1], values[-1]),
            xytext=(12, 0),
            textcoords="offset points",
            color=color,
            fontsize=10.3,
            weight="bold",
            va="center",
            clip_on=False
        )

    ax.set_xlabel(
        "Spatial points processed",
        color=MUTED,
        labelpad=15
    )

    ax.set_ylabel(
        "Execution time (s)  ·  logarithmic scale",
        color=MUTED,
        labelpad=13
    )

    fig.text(
        .075, .671,
        "MEASURED RUNTIME SCALING",
        color=MUTED,
        fontsize=10,
        weight="bold"
    )

    caption(
        fig,
        "GPU computation scales efficiently; "
        "total runtime increasingly reflects host I/O.",
        "performance_summary.json"
    )

    save(fig, "06_workload_scaling.png")




def plot_pipeline_components():
    """CPU reference versus GPU pipeline decomposition."""

    d = sorted(
        records("pipeline_breakdown.json"),
        key=lambda r: r["npoints"]
    )

    x = np.arange(len(d))
    labels = [f"{r['npoints']:,}" for r in d]

    # GPU pipeline components
    keys = [
        ("host_read_seconds", "Host I/O", ORANGE),
        ("transfer_to_device_seconds", "CPU → GPU", PURPLE),
        ("gpu_compute_seconds", "GPU compute", BLUE),
        ("transfer_from_device_seconds", "GPU → CPU", TEAL),
    ]

    fig = frame(
        "Anatomy of the processing pipeline",
        "Measured GPU pipeline stages across increasing spatial workloads",
        3
    )

    ax = axis(
        fig,
        bounds=(.13, .275, .72, .335)
    )

    # CPU computation reference
    perf = {
        r["npoints"]: r
        for r in records("performance_summary.json")
        if r["scenario"] == "performance_scaling"
    }

    cpu = np.array([
        perf[r["npoints"]]["cpu_compute_seconds_mean"]
        for r in d
    ])

    # CPU reference bars
    ax.bar(
        x - .20,
        cpu,
        width=.34,
        color=LIGHT,
        label="CPU compute (reference)",
        zorder=3
    )

    # Stacked GPU pipeline
    bottom = np.zeros(len(d))

    for key, label, color in keys:

        values = np.array([
            r[key] for r in d
        ])

        ax.bar(
            x + .20,
            values,
            bottom=bottom,
            width=.34,
            color=color,
            label=label,
            zorder=3
        )

        bottom += values

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(
        "Execution time (s)",
        color=MUTED,
        labelpad=12
    )

    ax.set_xlabel(
        "Spatial points processed",
        color=MUTED,
        labelpad=14
    )

    ax.set_ylim(
        0,
        max(max(bottom), max(cpu)) * 1.22
    )

    # Legend
    ax.legend(
        handles=[
            Patch(color=LIGHT, label="CPU compute")
        ] + [
            Patch(color=color, label=label)
            for _, label, color in keys
        ],
        loc="upper left",
        bbox_to_anchor=(0, 1.19),
        ncol=5,
        frameon=False,
        fontsize=9.1,
        columnspacing=1.1
    )

    # Annotations
    fig.text(
        .075, .670,
        "GPU PIPELINE BREAKDOWN",
        color=MUTED,
        fontsize=10,
        weight="bold"
    )

    fig.text(
        .75, .670,
        "LEFT: CPU  /  RIGHT: GPU pipeline",
        color=MUTED,
        fontsize=9.4,
        ha="right"
    )

    caption(
        fig,
        "Host I/O dominates GPU runtime; "
        "CPU compute is shown separately as a reference.",
        "pipeline_breakdown.json + performance_summary.json"
    )

    save(fig, "03_pipeline_breakdown.png")


def plot_vram_scaling():
    """
    Plot GPU VRAM consumption versus spatial batch size.

    Source:
        memory_scaling.json

    Scenario:
        batch_scaling

    Configuration:
        22,083 spatial points
        float32
        68,016 hourly observations per point
    """

    # --------------------------------------------------
    # 1. Load benchmark data
    # --------------------------------------------------

    d = [
        r for r in records("memory_scaling.json")
        if r["scenario"] == "batch_scaling"
        and r["dtype"] == "float32"
    ]

    if not d:
        raise ValueError(
            "No float32 batch_scaling records found "
            "in memory_scaling.json"
        )

    d = sorted(
        d,
        key=lambda r: r["actual_batch_size_mean"]
    )

    fixed = [
        r for r in d
        if r["requested_batch_size"] is not None
    ]

    auto = [
        r for r in d
        if r["requested_batch_size"] is None
    ]

    batch = np.array(
        [r["actual_batch_size_mean"] for r in fixed],
        dtype=float
    )

    vram = np.array(
        [r["peak_vram_gib"] for r in fixed],
        dtype=float
    )

    pool = np.array(
        [r["pool_used_gib"] for r in fixed],
        dtype=float
    )

    capacity = 7.61  # RTX 4070 Laptop GPU (GiB)

    # --------------------------------------------------
    # 2. Figure
    # --------------------------------------------------

    fig = frame(
        "GPU memory scaling",
        "Peak VRAM consumption versus spatial batch size"
        "  ·  22,083 spatial points",
        7
    )

    ax = axis(
        fig,
        bounds=(.14, .275, .69, .335)
    )

    # --------------------------------------------------
    # 3. Fixed batch measurements
    # --------------------------------------------------

    ax.plot(
        batch,
        vram,
        color=BLUE,
        lw=3.2,
        marker="o",
        ms=8,
        mfc="white",
        mew=2.2,
        zorder=5,
        label="Peak VRAM"
    )

    ax.plot(
        batch,
        pool,
        color=TEAL,
        lw=2.5,
        linestyle="--",
        marker="s",
        ms=6,
        mfc="white",
        mew=1.8,
        zorder=4,
        label="CuPy memory pool"
    )

    # --------------------------------------------------
    # 4. Adaptive batch measurement
    # --------------------------------------------------

    if auto:

        r = auto[0]

        auto_batch = float(
            r["actual_batch_size_mean"]
        )

        auto_vram = float(
            r["peak_vram_gib"]
        )

        auto_pool = float(
            r["pool_used_gib"]
        )

        # Highlight adaptive batch
        ax.scatter(
            [auto_batch],
            [auto_vram],
            s=155,
            marker="D",
            color=ORANGE,
            edgecolor="white",
            linewidth=1.8,
            zorder=8,
            label="Adaptive batch"
        )

        ax.scatter(
            [auto_batch],
            [auto_pool],
            s=65,
            marker="s",
            color=TEAL,
            edgecolor="white",
            linewidth=1.3,
            zorder=7
        )

        # Adaptive annotation:
        # placed to the right and below the point
        ax.annotate(
            f"AUTO  ·  {auto_batch:,.0f} points\n"
            f"{auto_vram:.2f} GiB",
            xy=(auto_batch, auto_vram),
            xytext=(48, -48),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            color=ORANGE,
            weight="bold",
            arrowprops=dict(
                arrowstyle="-",
                color=ORANGE,
                lw=1.1,
                connectionstyle="angle,angleA=0,angleB=-90,rad=8"
            ),
            zorder=10,
            annotation_clip=False
        )

    # --------------------------------------------------
    # 5. Physical GPU memory capacity
    # --------------------------------------------------

    ax.axhline(
        capacity,
        color=ORANGE,
        lw=1.6,
        linestyle=":",
        alpha=.85,
        zorder=2
    )

    ax.annotate(
        f"GPU capacity  ·  {capacity:.2f} GiB",
        xy=(2450, capacity),
        xytext=(0, 11),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=ORANGE,
        weight="bold"
    )

    # --------------------------------------------------
    # 6. Fixed-batch VRAM annotations
    # --------------------------------------------------

    for bx, vy in zip(batch, vram):

        if bx >= 1900:
            # Separate the 1,962-point measurement
            # from the adaptive 2,015-point measurement.
            offset = (-28, 24)
            ha = "right"

        elif bx == 1024:
            offset = (0, 18)
            ha = "center"

        elif bx == 512:
            offset = (0, 18)
            ha = "center"

        elif bx == 256:
            offset = (12, -20)
            ha = "left"

        else:
            offset = (-8, 18)
            ha = "center"

        ax.annotate(
            f"{vy:.2f} GiB",
            xy=(bx, vy),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=9.1,
            color=NAVY,
            weight="medium",
            zorder=9
        )

    # --------------------------------------------------
    # 7. Axes
    # --------------------------------------------------

    ax.set_xlim(0, 2650)
    ax.set_ylim(0, 8.8)

    # Use clean ticks instead of overlapping
    # 1,962 and 2,015 labels.
    ax.set_xticks([
        0, 256, 512, 1024, 1536, 2048, 2560
    ])

    ax.set_xticklabels([
        "0", "256", "512", "1,024",
        "1,536", "2,048", "2,560"
    ])

    ax.set_xlabel(
        "Spatial points per GPU batch",
        color=MUTED,
        labelpad=15
    )

    ax.set_ylabel(
        "GPU memory consumption (GiB)",
        color=MUTED,
        labelpad=13
    )

    ax.grid(
        axis="y",
        color=BORDER,
        lw=.8,
        alpha=.85
    )

    # --------------------------------------------------
    # 8. Legend
    # --------------------------------------------------

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.16),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.5,
        handlelength=2.2
    )

    # --------------------------------------------------
    # 9. Figure annotations
    # --------------------------------------------------

    fig.text(
        .075, .670,
        "GPU MEMORY SCALING",
        color=MUTED,
        fontsize=10,
        weight="bold"
    )

    caption(
        fig,
        "Adaptive batching processes a 5.60 GiB logical dataset "
        "with approximately 6.11 GiB peak VRAM.",
        "memory_scaling.json"
    )

    # --------------------------------------------------
    # 10. Save
    # --------------------------------------------------

    save(
        fig,
        "07_vram_scaling.png"
    )




def plot_performance_dashboard():
    """
    CUDAHeat performance dashboard.

    Combines:
      - performance_summary.json
      - pipeline_breakdown.json
      - throughput_summary.json
      - memory_scaling.json

    Output:
      Refined plots/08_performance_dashboard.png

    Uses existing FIG_W, FIG_H, DPI, records(), and save().
    """

    # ==================================================
    # 1. LOAD DATA
    # ==================================================

    perf = sorted(
        [
            r for r in records("performance_summary.json")
            if r["scenario"] == "performance_scaling"
        ],
        key=lambda r: r["npoints"]
    )

    pipeline = sorted(
        records("pipeline_breakdown.json"),
        key=lambda r: r["npoints"]
    )

    throughput = sorted(
        records("throughput_summary.json"),
        key=lambda r: r["npoints"]
    )

    memory = records("memory_scaling.json")

    batch_memory = sorted(
        [
            r for r in memory
            if r["scenario"] == "batch_scaling"
            and r["dtype"] == "float32"
            and r["requested_batch_size"] is not None
        ],
        key=lambda r: r["actual_batch_size_mean"]
    )

    adaptive_memory = [
        r for r in memory
        if r["scenario"] == "batch_scaling"
        and r["dtype"] == "float32"
        and r["requested_batch_size"] is None
    ]

    precision_memory = {
        r["dtype"]: r
        for r in memory
        if r["scenario"] == "precision_performance"
    }

    x = np.array([r["npoints"] for r in perf])

    cpu = np.array([
        r["cpu_compute_seconds_mean"] for r in perf
    ])

    gpu = np.array([
        r["gpu_compute_seconds_mean"] for r in perf
    ])

    wall = np.array([
        r["pipeline_wall_seconds_mean"] for r in perf
    ])

    compute_speedup = np.array([
        r["compute_speedup"] for r in throughput
    ])

    pipeline_speedup = np.array([
        r["pipeline_speedup"] for r in throughput
    ])

    batch = np.array([
        r["actual_batch_size_mean"] for r in perf
    ])

    # Largest workload
    largest = perf[-1]
    largest_pipeline = pipeline[-1]

    npoints = largest["npoints"]
    nhours = 68016

    total_values = npoints * nhours

    speedup = largest["pipeline_speedup_mean"]

    # ==================================================
    # 2. FIGURE AND LAYOUT
    # ==================================================

    fig = plt.figure(
        figsize=(FIG_W, FIG_H),
        dpi=DPI,
        facecolor=BG
    )

    fig.text(
        .045, .951,
        "CUDAHeat  /  PERFORMANCE ENGINEERING",
        fontsize=10,
        color=TEAL,
        weight="bold"
    )

    fig.text(
        .045, .902,
        "Inside the GPU pipeline",
        fontsize=26,
        color=INK,
        weight="bold"
    )

    fig.text(
        .046, .866,
        "Compute acceleration, memory efficiency and "
        "end-to-end performance",
        fontsize=10.5,
        color=MUTED
    )

    # ==================================================
    # 3. HELPER FUNCTIONS
    # ==================================================

    def card(x, y, w, h):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=(
                "round,pad=0.005,"
                "rounding_size=0.012"
            ),
            transform=fig.transFigure,
            facecolor=WHITE,
            edgecolor=BORDER,
            linewidth=.8,
            zorder=0
        )

        fig.add_artist(patch)

    def panel_axis(bounds):
        ax = fig.add_axes(
            bounds,
            facecolor=WHITE,
            zorder=2
        )

        ax.set_axisbelow(True)

        ax.grid(
            axis="y",
            color=BORDER,
            lw=.7,
            alpha=.8
        )

        ax.tick_params(
            axis="both",
            length=0,
            pad=5,
            labelcolor=MUTED,
            labelsize=7.4
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        return ax

    def panel_title(x, y, title, subtitle):
        fig.text(
            x, y,
            title,
            fontsize=11,
            color=INK,
            weight="bold"
        )

        fig.text(
            x, y - .024,
            subtitle,
            fontsize=7.6,
            color=MUTED
        )

    # ==================================================
    # 4. KPI CARDS
    # ==================================================

    kpi_y = .727
    kpi_h = .102
    kpi_w = .213

    kpis = [
        (
            "SPATIAL WORKLOAD",
            f"{npoints:,}",
            "valid ocean points",
            BLUE
        ),
        (
            "HOURLY SST VALUES",
            f"{total_values / 1e9:.2f}B",
            f"{nhours:,} hours per point",
            TEAL
        ),
        (
            "COMPUTE SPEEDUP",
            f"{compute_speedup[-1]:.1f}×",
            "GPU versus CPU",
            PURPLE
        ),
        (
            "PIPELINE SPEEDUP",
            f"{speedup:.1f}×",
            "including I/O",
            ORANGE
        )
    ]

    for i, (label, value, detail, color) in enumerate(kpis):

        xx = .045 + i * .232

        card(xx, kpi_y, kpi_w, kpi_h)

        fig.text(
            xx + .015,
            kpi_y + .075,
            label,
            fontsize=8.1,
            color=MUTED,
            weight="bold"
        )

        fig.text(
            xx + .015,
            kpi_y + .029,
            value,
            fontsize=22,
            color=color,
            weight="bold"
        )

        fig.text(
            xx + .120,
            kpi_y + .033,
            detail,
            fontsize=7.5,
            color=MUTED
        )

    # ==================================================
    # 5. DASHBOARD PANEL GEOMETRY
    # ==================================================

    # Two columns, three rows.
    left_x = .045
    right_x = .522

    card_w = .433
    card_h = .187

    row_y = [.510, .306, .102]

    for yy in row_y:
        card(left_x, yy, card_w, card_h)
        card(right_x, yy, card_w, card_h)

    # ==================================================
    # 6. PANEL A — EXECUTION TIME SCALING
    # ==================================================

    panel_title(
        .062, .665,
        "01  Execution time scaling",
        "CPU compute versus GPU compute and complete pipeline"
    )

    ax = panel_axis((.095, .542, .348, .080))

    ax.set_xscale("log")
    ax.set_yscale("log")

    for values, color, label in (
        (cpu, PURPLE, "CPU"),
        (wall, ORANGE, "Pipeline"),
        (gpu, BLUE, "GPU")
    ):
        ax.plot(
            x,
            values,
            color=color,
            lw=2,
            marker="o",
            ms=3.6,
            label=label
        )

    ax.set_xlim(x[0] * .8, x[-1] * 1.2)

    ax.legend(
        loc="upper left",
        fontsize=6.8,
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0, 1.12)
    )

    ax.set_xlabel(
        "Spatial points",
        fontsize=7,
        color=MUTED
    )

    ax.set_ylabel(
        "Seconds",
        fontsize=7,
        color=MUTED
    )

    # ==================================================
    # 7. PANEL B — PIPELINE DECOMPOSITION
    # ==================================================

    panel_title(
        .539, .665,
        "02  Where the time goes",
        "Measured components of the largest GPU workload"
    )

    stages = [
        ("Host I/O", "host_read_seconds", ORANGE),
        ("CPU → GPU", "transfer_to_device_seconds", PURPLE),
        ("GPU compute", "gpu_compute_seconds", BLUE),
        ("GPU → CPU", "transfer_from_device_seconds", TEAL)
    ]

    values = np.array([
        largest_pipeline[key]
        for _, key, _ in stages
    ])

    total_stages = values.sum()

    ax = panel_axis((.552, .566, .380, .055))

    left = 0

    for (label, _, color), value in zip(stages, values):

        ax.barh(
            0,
            value,
            left=left,
            height=.52,
            color=color,
            edgecolor=WHITE,
            linewidth=.8
        )

        left += value

    ax.set_xlim(0, total_stages * 1.02)
    ax.set_yticks([])
    ax.set_xlabel(
        "Execution time (s)",
        fontsize=7,
        color=MUTED
    )

    # Compact legend below the bar
    legend_y = .538

    for i, (label, _, color) in enumerate(stages):

        xx = .552 + i * .096

        fig.text(
            xx,
            legend_y,
            f"● {label}",
            fontsize=7.0,
            color=color,
            weight="bold"
        )

    # ==================================================
    # 8. PANEL C — SPEEDUP
    # ==================================================

    panel_title(
        .062, .461,
        "03  Acceleration",
        "Compute-only versus end-to-end speedup"
    )

    ax = panel_axis((.095, .339, .348, .080))

    speed_x = np.array([
        r["npoints"] for r in throughput
    ])

    ax.set_xscale("log")

    ax.plot(
        speed_x,
        compute_speedup,
        color=BLUE,
        lw=2.2,
        marker="o",
        ms=3.8,
        label="Compute"
    )

    ax.plot(
        speed_x,
        pipeline_speedup,
        color=ORANGE,
        lw=2.2,
        marker="o",
        ms=3.8,
        label="Pipeline"
    )

    ax.fill_between(
        speed_x,
        pipeline_speedup,
        compute_speedup,
        color=BLUE,
        alpha=.06
    )

    ax.set_xlim(speed_x[0] * .8, speed_x[-1] * 1.2)
    ax.set_ylim(0, max(compute_speedup) * 1.15)

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=7
    )

    ax.set_xlabel(
        "Spatial points",
        fontsize=7,
        color=MUTED
    )

    ax.set_ylabel(
        "Speedup (×)",
        fontsize=7,
        color=MUTED
    )

    # ==================================================
    # 9. PANEL D — VRAM SCALING
    # ==================================================

    panel_title(
        .539, .461,
        "04  Memory scaling",
        "Measured peak VRAM versus spatial batch size"
    )

    bx = np.array([
        r["actual_batch_size_mean"]
        for r in batch_memory
    ])

    bv = np.array([
        r["peak_vram_gib"]
        for r in batch_memory
    ])

    ax = panel_axis((.572, .339, .348, .080))

    ax.plot(
        bx,
        bv,
        color=BLUE,
        lw=2.2,
        marker="o",
        ms=4,
        label="Fixed batch"
    )

    if adaptive_memory:

        ar = adaptive_memory[0]

        ax.scatter(
            [ar["actual_batch_size_mean"]],
            [ar["peak_vram_gib"]],
            marker="D",
            s=48,
            color=ORANGE,
            edgecolor=WHITE,
            linewidth=1,
            zorder=5,
            label="Adaptive"
        )

    ax.axhline(
        7.61,
        color=ORANGE,
        linestyle=":",
        lw=1.2
    )

    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 8.5)

    ax.legend(
        loc="upper left",
        fontsize=6.8,
        frameon=False
    )

    ax.set_xlabel(
        "Points per batch",
        fontsize=7,
        color=MUTED
    )

    ax.set_ylabel(
        "Peak VRAM (GiB)",
        fontsize=7,
        color=MUTED
    )

    # ==================================================
    # 10. PANEL E — ADAPTIVE BATCHING
    # ==================================================

    panel_title(
        .062, .257,
        "05  Adaptive batching",
        "Batch width versus total spatial workload"
    )

    ax = panel_axis((.095, .135, .348, .080))

    ax.set_xscale("log")

    ax.plot(
        x,
        batch,
        color=TEAL,
        lw=2.2,
        marker="o",
        ms=4
    )

    ax.fill_between(
        x,
        0,
        batch,
        color=TEAL,
        alpha=.07
    )

    ax.set_xlim(x[0] * .8, x[-1] * 1.2)
    ax.set_ylim(0, max(batch) * 1.2)

    ax.set_xlabel(
        "Total spatial points",
        fontsize=7,
        color=MUTED
    )

    ax.set_ylabel(
        "Points per batch",
        fontsize=7,
        color=MUTED
    )

    # ==================================================
    # 11. PANEL F — FLOAT32 VERSUS FLOAT64
    # ==================================================

    panel_title(
        .539, .257,
        "06  Numerical precision",
        "Memory footprint and adaptive batch width"
    )

    if {"float32", "float64"} <= precision_memory.keys():

        p32 = precision_memory["float32"]
        p64 = precision_memory["float64"]

        px = np.arange(2)

        logical = [
            p32["logical_input_gib"],
            p64["logical_input_gib"]
        ]

        peak = [
            p32["peak_vram_gib"],
            p64["peak_vram_gib"]
        ]

        ax = panel_axis((.572, .135, .348, .080))

        ax.bar(
            px - .17,
            logical,
            width=.32,
            color=LIGHT,
            label="Logical input"
        )

        ax.bar(
            px + .17,
            peak,
            width=.32,
            color=[BLUE, PURPLE],
            label="Peak VRAM"
        )

        ax.set_xticks(px)
        ax.set_xticklabels(["float32", "float64"])

        ax.set_ylim(
            0,
            max(logical + peak) * 1.22
        )

        ax.set_ylabel(
            "Memory (GiB)",
            fontsize=7,
            color=MUTED
        )

        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=6.7,
            ncol=2
        )

        # Show actual adaptive batch widths.
        for i, record in enumerate((p32, p64)):

            ax.text(
                i,
                max(
                    record["logical_input_gib"],
                    record["peak_vram_gib"]
                ) + .45,
                f'{record["actual_batch_size_mean"]:,.0f} pts/batch',
                ha="center",
                fontsize=6.8,
                color=INK
            )

    # ==================================================
    # 12. FOOTER
    # ==================================================

    fig.text(
        .045, .050,
        "CUDAHeat  /  Full-pipeline presentation benchmark",
        fontsize=8.1,
        color=INK,
        weight="bold"
    )

    fig.text(
        .045, .027,
        "Measured results  ·  float32 unless specified  ·  "
        "68,016 hourly observations per point",
        fontsize=7.4,
        color=MUTED
    )

    fig.text(
        .955, .035,
        "08  /  PERFORMANCE DASHBOARD",
        fontsize=8.0,
        color=TEAL,
        weight="bold",
        ha="right"
    )

    # ==================================================
    # 13. SAVE
    # ==================================================

    save(
        fig,
        "08_performance_dashboard.png"
    )






PLOT_REGISTRY = {
    'gpu_acceleration': plot_gpu_acceleration,
    'batch_shape': plot_batch_shape,
    'pipeline_components': plot_pipeline_components,
    'precision_comparison': plot_precision_comparison,
    'pipeline_speedup': plot_pipeline_speedup,
    "workload_scaling": plot_workload_scaling,
    "performance_dashboard": plot_performance_dashboard,
}
PLOT_REGISTRY["pipeline_components"] = plot_pipeline_components
PLOT_REGISTRY["vram_scaling"] = plot_vram_scaling

ENABLED_PLOTS = tuple(PLOT_REGISTRY)  # Remove keys here to generate only selected figures.


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    for name in ENABLED_PLOTS:
        PLOT_REGISTRY[name]()

if __name__=='__main__':main()