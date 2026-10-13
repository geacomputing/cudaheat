#!/usr/bin/env python3
"""CUDAHeat NetCDF I/O diagnostics — 16:9 slide at 500 DPI.
Numbers transcribed from the user's 2026-10-09 NetCDF inspection JSON.
This slide shows ORIGINAL layout diagnostics and a PROPOSED optimization,
not a measured before/after comparison.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle

DPI = 500
OUT = Path(__file__).resolve().parent / 'cudaheat_netcdf_chunking_diagnostics.png'
BG='#F7F9FB'; INK='#17283B'; MUTED='#617287'; BLUE='#2475B6'; TEAL='#087F8C'; ORANGE='#E36B45'; GRID='#DFE7EE'; WHITE='#FFFFFF'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.facecolor':BG})
fig=plt.figure(figsize=(40/3,7.5),dpi=DPI,facecolor=BG)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

def txt(x,y,s,size=10,color=INK,weight='normal',ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,transform=ax.transAxes,**kw)
def box(x,y,w,h,edge=GRID,lw=.8,fc=WHITE,r=.014):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.003,rounding_size={r}',transform=ax.transAxes,fc=fc,ec=edge,lw=lw)
    ax.add_patch(p); return p
def line(x1,y1,x2,y2,c=GRID,lw=1,ls='-'):
    ax.plot([x1,x2],[y1,y2],transform=ax.transAxes,color=c,lw=lw,ls=ls,solid_capstyle='round')
def arrow(x1,y1,x2,y2,c=TEAL,lw=1.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),transform=ax.transAxes,arrowstyle='-|>',mutation_scale=12,lw=lw,color=c))

# Header
text_top=.929
txt(.052,text_top,'NETCDF I/O  /  THE CHUNKING BOTTLENECK',19,INK,'bold')
txt(.052,.878,'Original storage diagnostics  •  Why CUDAHeat needs spatially aligned, full-time reads',10.5,MUTED)

# Top KPI row
kpis=[('68,016','HOURLY STEPS'),('41,377','SPATIAL POINTS'),('25,056','PHYSICAL CHUNKS'),('32','TIME CHUNKS / SERIES')]
left=.052; gap=.015; width=(.896-3*gap)/4
for i,(v,label) in enumerate(kpis):
    x=left+i*(width+gap)
    box(x,.752,width,.088,edge=GRID)
    txt(x+.015,.804,v,17,BLUE if i<2 else TEAL,'bold')
    txt(x+.015,.769,label,7.5,MUTED,'bold')

# Main left chart
box(.052,.286,.523,.428)
txt(.075,.682,'MEASURED READ LATENCY',12,INK,'bold')
txt(.075,.651,'Median of 10 repeated reads  ·  original NetCDF',8.5,MUTED)
labels=['257 contiguous points · full time','257 strided points · full time','16 × 16 tile · full time','257 points · decoded float64']
values=np.array([.592150,.599378,.074554,.686355]); colors=[BLUE,'#7092B0',TEAL,ORANGE]
# Bars aligned; physical height corrected automatically by using axis coords for positions only
start=.595; step=.069; bx=.285; bmax=.235
for i,(label,val,col) in enumerate(zip(labels,values,colors)):
    y=start-i*step
    txt(.075,y,label,8.1,INK)
    ax.add_patch(Rectangle((bx,y-.012),bmax,.024,transform=ax.transAxes,fc='#EFF3F7',ec='none'))
    ax.add_patch(Rectangle((bx,y-.012),bmax*(val/.75),.024,transform=ax.transAxes,fc=col,ec='none'))
    txt(.535,y,f'{val*1000:.1f} ms',8.5,INK,'bold',ha='right')
txt(.075,.331,'The 16 × 16 tile reads ~7.9× faster than the contiguous 257-point strip',9.1,TEAL,'bold')
txt(.075,.305,'for comparable raw output sizes (~33 MiB); access geometry matters.',8.2,MUTED)

# Right schematic
box(.597,.286,.351,.428)
txt(.62,.682,'PHYSICAL LAYOUT vs. GPU NEED',11.6,INK,'bold')
txt(.62,.65,'Current chunk  (time × lat × lon)',8.4,MUTED)
txt(.62,.614,'2,126 × 6 × 9',15.2,ORANGE,'bold')
# time segmented strip
x0=.622; y0=.562; ww=.300; hh=.044
for j in range(16):
    ax.add_patch(Rectangle((x0+j*ww/16,y0),ww/16-.001,hh,transform=ax.transAxes,
                           fc='#F5C6B7' if j%2==0 else '#FBE3DC',ec=WHITE,lw=.45))
txt(.622,.538,'Time split into 32 physical chunks per complete series',8.2,MUTED)
arrow(.775,.513,.775,.474,c=TEAL)
txt(.62,.444,'TARGET: FULL TIME × SPATIAL BATCH',9.4,TEAL,'bold')
# uninterrupted time stripe, with vertical stripes indicating spatial points
ax.add_patch(Rectangle((.622,.375),.300,.045,transform=ax.transAxes,fc='#E4F3F2',ec=TEAL,lw=1.0))
for j in range(1,15):
    line(.622+j*.300/15,.375,.622+j*.300/15,.420,'#B5DCD8',.65)
txt(.622,.349,'68,016 hours intact  •  partition space, never time',8.4,INK)
txt(.622,.315,'Illustrative target layout; not yet benchmarked here.',7.6,MUTED)

# Footer callout
box(.052,.122,.896,.123,edge='#C8DCD9',lw=1.0,fc='#F0F8F7')
txt(.075,.210,'OPTIMIZATION STRATEGY',9,TEAL,'bold')
txt(.075,.175,'Reorganize SST as (time, point), remove permanently invalid land points,',10.1,INK,'bold')
txt(.075,.148,'and physically chunk by full time × contiguous spatial batches to reduce read amplification.',9.4,INK)
txt(.052,.075,'METHOD NOTE  •  Repeated reads may benefit from OS/HDF5 caching; timings are not independent cold-cache trials.',8.2,MUTED)
#txt(.948,.075,'SOURCE  /  NETCDF INSPECTION  ·  09 OCT 2026',7.8,MUTED,ha='right')
fig.savefig(OUT,dpi=DPI,bbox_inches=None,pad_inches=0)
#plt.close(fig)
print(f'Saved {OUT} (6667 × 3750 pixels)')