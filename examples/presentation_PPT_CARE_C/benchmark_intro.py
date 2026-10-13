#!/usr/bin/env python3
"""CUDAHeat benchmark introduction — 16:9 PowerPoint slide at 500 DPI."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D

DPI = 500
OUT = Path(__file__).resolve().parent
BG='#F7F9FB'; WHITE='#FFFFFF'; INK='#17283B'; MUTED='#617287'
BLUE='#2475B6'; TEAL='#087F8C'; ORANGE='#E36B45'; GRID='#E1E8EE'; PURPLE='#8071AC'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.facecolor':BG})
fig=plt.figure(figsize=(40/3,7.5),dpi=DPI,facecolor=BG)
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
def txt(x,y,s,size=10,color=INK,weight='normal',ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,transform=ax.transAxes,**kw)
def box(x,y,w,h,fc=WHITE,ec=GRID,lw=.85,r=.012):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.001,rounding_size={r}',fc=fc,ec=ec,lw=lw,transform=ax.transAxes)
    ax.add_patch(p);return p

txt(.052,.934,'CUDAHeat  /  PERFORMANCE EVALUATION',9,TEAL,'bold')
txt(.052,.876,'Benchmarking at scale',23,INK,'bold')
txt(.052,.824,'Computational efficiency, memory behavior, and scalability of hourly MHW detection',10.3,MUTED)

# Dataset banner
box(.052,.622,.896,.155)
txt(.072,.745,'BENCHMARK DATASET',8.4,TEAL,'bold')
metrics=[('41,377','SPATIAL GRID POINTS','257 × 161'),('~8 years','HOURLY SST RECORD','continuous; no gaps'),('~2.9 B','TEMPERATURE VALUES','~70,000 hourly steps'),('11.6 GB','FLOAT32 INPUT','~23.2 GB in float64')]
for j,(big,lab,detail) in enumerate(metrics):
    x=.073+j*.219
    if j:
        ax.add_line(Line2D([x-.015,x-.015],[.647,.742],lw=.9,color=GRID,transform=ax.transAxes))
    txt(x,.703,big,17.5,INK,'bold')
    txt(x,.669,lab,7.4,MUTED,'bold')
    txt(x,.644,detail,7.8,MUTED)

# Five benchmark categories in one line
head_y=.566
txt(.052,head_y,'EXPERIMENTAL ASSESSMENT',9,INK,'bold')
txt(.948,head_y,'PARAMETER SWEEPS  ·  DATA CONFIGURATIONS  ·  EXECUTION SETTINGS',7.4,MUTED,ha='right')
items=[
    ('01','MEMORY','RAM / VRAM usage\nadaptive batching',BLUE),
    ('02','PERFORMANCE','Execution time\nthroughput & speedup',TEAL),
    ('03','PIPELINE COST','NetCDF I/O → transfers\nGPU compute → output',PURPLE),
    ('04','SCALING','Dataset size\nbatch size & time span',BLUE),
    ('05','ROBUSTNESS','float32 vs float64\nparameter sensitivity',ORANGE),
]
gap=.012; left=.052; total=.896; w=(total-gap*4)/5
for i,(num,title,body,color) in enumerate(items):
    x=left+i*(w+gap)
    box(x,.308,w,.220)
    ax.add_patch(Rectangle((x+.014,.499),.037,.0045,fc=color,ec='none',transform=ax.transAxes))
    txt(x+.015,.468,num,9.2,color,'bold')
    txt(x+.015,.418,title,9.4,INK,'bold')
    txt(x+.015,.356,body,8.15,MUTED,va='center',linespacing=1.65)

# Main question + next-slide handoff
box(.052,.132,.896,.121,fc='#EAF3F6',ec='#D4E5EA',lw=.8)
txt(.073,.220,'CORE QUESTION',8.0,TEAL,'bold')
txt(.073,.180,'How efficiently can GPUs process billions of hourly SST values',12.0,INK,'bold')
txt(.073,.151,'while preserving complete time series for robust MHW detection?',11.0,INK)

txt(.052,.075,'NEXT  →  Benchmark diagnostics, pipeline bottlenecks, speedup and scaling',9.3,MUTED)
txt(.948,.075,'COPERNICUS MARINE  ·  CUDAHeat',7.5,MUTED,ha='right')

fig.savefig(OUT/'cudaheat_benchmark_intro.png',dpi=DPI,bbox_inches=None,pad_inches=0)
fig.savefig(OUT/'cudaheat_benchmark_intro_preview.png',dpi=120,bbox_inches=None,pad_inches=0)
#plt.close(fig)
print('Saved slide and preview')