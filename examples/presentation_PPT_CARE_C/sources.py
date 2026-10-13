#!/usr/bin/env python3
"""CUDAHeat SST data sources — 16:9, 500 DPI, PowerPoint-ready."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D

DPI = 500
OUT = Path(__file__).resolve().parent / 'cudaheat_data_sources.png'
BG, WHITE, INK, MUTED = '#F7F9FB', '#FFFFFF', '#17283B', '#617287'
BLUE, TEAL, VIOLET, ORANGE = '#2475B6', '#087F8C', '#8071AC', '#D97750'
GRID, LIGHT = '#E1E8EE', '#EDF3F7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig = plt.figure(figsize=(40/3, 7.5), dpi=DPI, facecolor=BG)
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

def txt(x,y,s,size=9,color=INK,weight='normal',ha='left',va='center',**kwargs):
    return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,transform=ax.transAxes,**kwargs)
def line(x1,y1,x2,y2,c=GRID,lw=1,alpha=1):
    ax.add_line(Line2D([x1,x2],[y1,y2],transform=ax.transAxes,color=c,lw=lw,alpha=alpha,solid_capstyle='round'))
def panel(x,y,w,h,edge=GRID,lw=0.9,fill=WHITE):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.002,rounding_size=0.012',transform=ax.transAxes,fc=fill,ec=edge,lw=lw)
    ax.add_patch(p)
def item(x,y,s,color,positive=True):
    ax.add_patch(Circle((x,y),.0046,transform=ax.transAxes,fc=color,ec='none'))
    txt(x+.014,y,s,8.45,MUTED,va='center',linespacing=1.42)

txt(.055,.935,'HOURLY SST  /  DATA SOURCES',19,INK,'bold')
txt(.055,.891,'Three complementary routes to support hourly marine heatwave detection',10.3,MUTED)

xs=[.055,.359,.663]; w=.282; y=.145; h=.681
cols=[BLUE,TEAL,VIOLET]
for x,c in zip(xs,cols):
    panel(x,y,w,h)
    ax.add_patch(Rectangle((x+.011,y+h-.010),w-.022,.0045,transform=ax.transAxes,fc=c,ec='none'))

# Header badges
for i,(x,c) in enumerate(zip(xs,cols)):
    txt(x+.024,.787,f'0{i+1}',10,c,'bold')

# Card 1
x=xs[0]
txt(x+.024,.738,'COPERNICUS MARINE',12.3,INK,'bold')
txt(x+.024,.703,'Satellite-derived  /  L4 hourly SST',9,MUTED)
# heatmap schematic
xx=np.linspace(0,1,36); yy=np.linspace(0,1,10)
field=(np.sin(xx[None,:]*8)*.30+np.cos(yy[:,None]*7)*.14+xx[None,:]*.6)
mini=ax.inset_axes([x+.024,.528,w-.048,.125],transform=ax.transAxes)
mini.imshow(field,aspect='auto',cmap='YlGnBu',origin='lower',interpolation='bilinear')
mini.set_xticks([]);mini.set_yticks([])
for spine in mini.spines.values():spine.set_visible(False)
txt(x+.024,.510,'0.0625° grid  ·  ~6 km  ·  Mediterranean-wide',7.7,MUTED)
line(x+.024,.482,x+w-.024,.482)
txt(x+.024,.458,'ADVANTAGES',8.2,TEAL,'bold')
item(x+.029,.426,'Level-4, gap-free gridded analysis',TEAL)
item(x+.029,.393,'Easy, fast API retrieval',TEAL)
item(x+.029,.360,'High spatial resolution (~6 km)',TEAL)
item(x+.029,.327,'Excellent basin-wide coverage',TEAL)
line(x+.024,.298,x+w-.024,.298)
txt(x+.024,.275,'LIMITATIONS',8.2,ORANGE,'bold')
item(x+.029,.243,'Short record (available since 2019)',ORANGE)
item(x+.029,.207,'Model-assisted interpolation may\nintroduce uncertainty in hourly SST',ORANGE)

# Card 2
x=xs[1]
txt(x+.024,.738,'DRAGONERA BUOY',12.3,INK,'bold')
txt(x+.024,.703,'In situ  /  Spanish port authority',9,MUTED)
mini=ax.inset_axes([x+.024,.528,w-.048,.125],transform=ax.transAxes)
t=np.linspace(0,9,350)
sst=23+.25*np.sin(t*2.4)+.10*np.sin(t*9)+.10*np.cos(t*.8)
mask=((t>2.1)&(t<3.25))|((t>6.0)&(t<6.75))
sst[mask]=np.nan
mini.plot(t,sst,color=TEAL,lw=1.8)
mini.set_xlim(t.min(),t.max());mini.set_ylim(22.55,23.55)
mini.set_xticks([]);mini.set_yticks([])
for spine in mini.spines.values():spine.set_visible(False)
mini.set_facecolor(LIGHT)
txt(x+.024,.510,'High-frequency observations  ·  missing intervals',7.7,MUTED)
line(x+.024,.482,x+w-.024,.482)
txt(x+.024,.458,'ADVANTAGES',8.2,TEAL,'bold')
item(x+.029,.426,'Direct water-temperature measurement',TEAL)
item(x+.029,.393,'Sub-hourly sampling possible',TEAL)
item(x+.029,.360,'In situ reference (“ground truth”)',TEAL)
line(x+.024,.326,x+w-.024,.326)
txt(x+.024,.302,'LIMITATIONS',8.2,ORANGE,'bold')
item(x+.029,.270,'~16 years of available observations',ORANGE)
item(x+.029,.237,'Frequent gaps; some last months',ORANGE)
# subtle proposed solution
panel(x+.024,.169,w-.048,.044,edge='#D1E7E4',fill='#F1F9F7',lw=.8)
txt(x+.036,.191,'NEXT  /  ML-assisted gap reconstruction',8.1,TEAL,'bold')

# Card 3
x=xs[2]
txt(x+.024,.738,'REANALYSIS',12.3,INK,'bold')
txt(x+.024,.703,'Daily SST  /  Temporal downscaling',9,MUTED)
mini=ax.inset_axes([x+.024,.528,w-.048,.125],transform=ax.transAxes)
t=np.linspace(0,7,280)
coarse=22.8+.12*np.arange(8)+.18*np.sin(np.arange(8)*.9)
hourly=np.interp(t,np.arange(8),coarse)+.17*np.sin(2*np.pi*t)
mini.plot(np.arange(8),coarse,color=BLUE,lw=1.6,marker='o',ms=4,label='daily')
mini.plot(t,hourly,color=VIOLET,lw=1.5,alpha=.85,label='hourly')
mini.set_xlim(0,7);mini.set_ylim(22.5,24.1)
mini.set_xticks([]);mini.set_yticks([])
for spine in mini.spines.values():spine.set_visible(False)
mini.set_facecolor(LIGHT)
txt(x+.024,.510,'Daily baseline  →  reconstructed hourly cycle',7.7,MUTED)
line(x+.024,.482,x+w-.024,.482)
txt(x+.024,.454,'OBJECTIVE',8.2,VIOLET,'bold')
item(x+.029,.416,'Generate physically plausible\nhourly SST from daily fields',VIOLET)
item(x+.029,.364,'Extend temporal coverage beyond\nthe available hourly records',VIOLET)
line(x+.024,.311,x+w-.024,.311)
txt(x+.024,.285,'KEY CHALLENGE',8.2,ORANGE,'bold')
item(x+.029,.249,'Recover sub-daily variability without\ninventing spurious MHW events',ORANGE)
panel(x+.024,.169,w-.048,.044,edge='#E4DDF0',fill='#F7F4FC',lw=.8)
txt(x+.036,.191,'STATUS  /  Work in progress',8.1,VIOLET,'bold')

# Footer
line(.055,.115,.945,.115)
txt(.055,.087,'DATA STRATEGY',8.3,INK,'bold')
txt(.177,.087,'Spatial coverage  +  direct observations  +  longer reconstructed records',9.0,MUTED)
txt(.055,.051,'Conceptual overview  ·  Sources differ in spatial coverage, temporal completeness and observational independence.',8.0,MUTED)
fig.savefig(OUT,dpi=DPI,bbox_inches=None,pad_inches=0,facecolor=BG)
#plt.close(fig)
print(OUT)