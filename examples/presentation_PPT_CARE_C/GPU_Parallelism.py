#!/usr/bin/env python3
"""CUDAHeat GPU architecture infographic, 16:9, 500 dpi. Run: python cudaheat_gpu_parallelism.py"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

DPI=500
OUT=Path(__file__).resolve().parent/'cudaheat_gpu_parallelism.png'
BG='#F7F9FB'; WHITE='#FFFFFF'; INK='#17283B'; MUTED='#617287'; BLUE='#2475B6'; TEAL='#087F8C'; ORANGE='#E36B45'; GRID='#DDE6ED'; PURPLE='#8071AC'
fig=plt.figure(figsize=(6667/DPI,3750/DPI),dpi=DPI,facecolor=BG)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')

def text(x,y,s,sz=9,color=INK,weight='normal',ha='left',va='center',**kw):
    ax.text(x,y,s,fontsize=sz,color=color,weight=weight,ha=ha,va=va,transform=ax.transAxes,**kw)
def line(x1,y1,x2,y2,color=GRID,lw=1.2,alpha=1):
    ax.add_line(Line2D([x1,x2],[y1,y2],transform=ax.transAxes,color=color,lw=lw,alpha=alpha,solid_capstyle='round'))
def box(x,y,w,h,edge=GRID,lw=.9,fc=WHITE,r=.012):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.001,rounding_size={r}',transform=ax.transAxes,fc=fc,ec=edge,lw=lw);ax.add_patch(p);return p
def arrow(x1,y1,x2,y2,color=TEAL,lw=1.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),transform=ax.transAxes,arrowstyle='-|>',mutation_scale=12,lw=lw,color=color,connectionstyle='arc3'))

text(.055,.931,'WHY GPUs ACCELERATE HOURLY MHW DETECTION',19,INK,'bold')
text(.055,.887,'Many independent SST series  ·  one identical algorithm  ·  full temporal history per point',10.5,MUTED)

# top takeaway strip
box(.055,.794,.89,.060,edge='#BBDAD9',fc='#EAF5F4',r=.010)
text(.075,.824,'THE KEY IDEA',9,TEAL,'bold')
text(.185,.824,'Parallelise across spatial points; preserve time within every series.',12,INK,'bold')

# 3 panels
panels=[(.055,.333,.276),(.362,.333,.276),(.669,.333,.276)]
for x,y,w in panels:box(x,y,w,.424)

# Panel 1: data structure
x,y,w=panels[0]
text(x+.020,y+.391,'01  THE DATA',9,BLUE,'bold')
text(x+.020,y+.347,'A spatial batch',13,INK,'bold')
text(x+.020,y+.313,'B points × T hourly samples',9,MUTED)
# horizontal time lines, points rows
x0=x+.039; x1=x+w-.023
for j in range(7):
    yy=y+.255-j*.030
    line(x0,yy,x1,yy,GRID,1.2)
    for k in range(24):
        xx=x0+(x1-x0)*k/23
        c=TEAL if j in (1,3,5) else BLUE
        ax.add_patch(Circle((xx,yy),.0026,transform=ax.transAxes,fc=c,ec='none',alpha=.50+.38*(k%5)/4))
    text(x+.012,yy,f'p{j+1}',7.4,MUTED,ha='left')
text(x+.038,y+.034,'TIME →  complete and continuous',8.8,TEAL,'bold')
text(x+.038,y+.069,'Each row is an independent SST history',8.2,MUTED)

# Panel 2: GPU threads
x,y,w=panels[1]
text(x+.020,y+.391,'02  THE GPU',9,TEAL,'bold')
text(x+.020,y+.347,'Thousands of lightweight threads',11.3,INK,'bold')
text(x+.020,y+.314,'SMs schedule groups of 32 threads (warps)',8.6,MUTED)
# stylized GPU blocks
for rr in range(3):
    for cc in range(8):
        xx=x+.024+cc*.0285; yy=y+.235-rr*.050
        fc=TEAL if rr<2 else '#C6DDE2'
        ax.add_patch(FancyBboxPatch((xx,yy),.021,.033,boxstyle='round,pad=0.001,rounding_size=0.004',transform=ax.transAxes,fc=fc,ec='none',alpha=.88))
text(x+.023,y+.074,'Thread i  →  point i  →  all T hours',9.3,TEAL,'bold')
text(x+.023,y+.040,'Same operations; independent data',8.5,MUTED)

# Panel 3: time operations
x,y,w=panels[2]
text(x+.020,y+.391,'03  THE WORK',9,PURPLE,'bold')
text(x+.020,y+.347,'Repeated across every time series',11.3,INK,'bold')
text(x+.020,y+.312,'Threshold  →  gaps  →  duration  →  events',8.6,MUTED)
rng=np.random.default_rng(5)
t=np.linspace(0,1,230)
for j in range(4):
    yy=y+.250-j*.046
    vals=.011*np.sin(2*np.pi*(4*t+.22*j))+.008*np.sin(2*np.pi*(1.1*t+j*.17))+.012*t
    ax.plot(x+.027+t*(w-.054), yy+vals,color=[BLUE,TEAL,PURPLE,ORANGE][j],lw=1.45,transform=ax.transAxes)
    line(x+.027,yy+.005,x+w-.027,yy+.005,GRID,.7)
text(x+.020,y+.040,'Temporal order stays available',8.8,PURPLE,'bold')

# arrows between panels
arrow(.337,.553,.358,.553,TEAL,1.7)
arrow(.644,.553,.665,.553,TEAL,1.7)

# bottom flow strip
text(.055,.283,'WHY IT IS FAST',9.5,INK,'bold')
steps=[('MASSIVE PARALLELISM','Many points processed concurrently',BLUE),('SIMPLE REPEATED WORK','Similar instructions across points',TEAL),('FULL TIME SERIES','No temporal boundary stitching',PURPLE),('ADAPTIVE BATCHING','Fit VRAM; repeat for next batch',ORANGE)]
for i,(head,body,c) in enumerate(steps):
    xx=.055+i*.226
    box(xx,.148,.209,.105,edge=GRID,fc=WHITE,r=.008)
    ax.add_patch(Rectangle((xx+.013,.222),.029,.004,transform=ax.transAxes,fc=c,ec='none'))
    text(xx+.013,.201,head,8.5,c,'bold')
    text(xx+.013,.171,body,7.7,MUTED)

line(.055,.109,.945,.109,GRID,1)
text(.055,.081,'IMPORTANT',8.3,ORANGE,'bold')
text(.131,.081,'Preserving time is required for event logic; speed also depends on memory layout, coalescing and I/O.',8.5,MUTED)
text(.055,.044,'CUDAHEAT  /  CONCEPTUAL GPU EXECUTION MODEL',8,MUTED)
fig.savefig(OUT,dpi=DPI,bbox_inches=None,pad_inches=0,facecolor=BG)
#plt.close(fig)
print(f'Saved {OUT} (6667 × 3750 px)')