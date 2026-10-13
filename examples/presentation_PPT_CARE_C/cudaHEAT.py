#!/usr/bin/env python3
"""CUDAHeat spatial batching infographic, 16:9 PowerPoint, 500 DPI.

Run: python cudaheat_spatiotemporal_batching.py
Outputs a PNG next to this script. Only NumPy and Matplotlib required.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

DPI = 500
W, H = 6667 / 500, 7.5
OUT = Path(__file__).resolve().parent / 'cudaheat_spatiotemporal_batching.png'
BG, WHITE, INK, MUTED = '#F7F9FB', '#FFFFFF', '#17283B', '#617287'
BLUE, TEAL, ORANGE, PURPLE = '#2475B6', '#087F8C', '#E36B45', '#7567A4'
BORDER, SOFT, GREEN = '#DDE6ED', '#EDF4F8', '#228A77'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'savefig.facecolor': BG})
fig = plt.figure(figsize=(W,H), dpi=DPI, facecolor=BG)
ax = fig.add_axes([0,0,1,1], xlim=(0,W), ylim=(0,H), aspect='equal')
ax.axis('off')

def text(x,y,s,size=10,color=INK,weight='normal',ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,**kw)

def line(a,b,c=INK,lw=1.5,alpha=1,ls='-'):
    ax.add_line(Line2D([a[0],b[0]],[a[1],b[1]],lw=lw,color=c,alpha=alpha,ls=ls,solid_capstyle='round'))

def box(x,y,w,h,fill=WHITE,edge=BORDER,lw=.9,r=.13):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fill,edgecolor=edge,lw=lw)
    ax.add_patch(p); return p

def arrow(a,b,color=TEAL,lw=1.7,style='-|>',ms=12,rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=ms,lw=lw,color=color,
                                 connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0))

def pill(x,y,w,label,fill=SOFT,color=BLUE):
    box(x,y,w,.28,fill,fill,r=.14)
    text(x+w/2,y+.14,label,7.1,color,'bold',ha='center')

# Heading
text(.56,7.02,'CUDAHEAT  /  SPATIOTEMPORAL GPU PROCESSING',19,INK,'bold')
text(.57,6.62,'Keep the complete hourly history on each GPU batch  ·  partition only the spatial domain',10.3,MUTED)
#pill(11.10,6.87,1.60,'500 DPI  ·  16:9',fill='#E9F0F5',color=MUTED)

# Upper 3-stage strip
strip_y, strip_h = 4.87,1.39
blocks=[(.56,3.81,'01','NETCDF LAYOUT','Spatial chunks  ·  full time'),
        (4.54,3.81,'02','HOST → GPU BATCHES','Stream point blocks to VRAM'),
        (8.52,4.25,'03','PRECISION','float64 → float32  ·  validated')]
for x,w,n,title,subtitle in blocks:
    box(x,strip_y,w,strip_h)
    text(x+.20,strip_y+strip_h-.29,n,8.5,TEAL,'bold')
    text(x+.61,strip_y+strip_h-.29,title,10.4,INK,'bold')
    text(x+.20,strip_y+.22,subtitle,8.4,MUTED)

# netcdf mini grid: horizontal full-length strips (time runs left-right)
x0,y0=1.00,5.26
for row in range(3):
    for col in range(8):
        ax.add_patch(Rectangle((x0+col*.29,y0+row*.12),.275,.102,
                    facecolor=[BLUE,TEAL,'#86AEC4'][row],alpha=.77,edgecolor='white',lw=.45))
text(3.46,5.46,'t →',7.4,BLUE,'bold')
# GPU batches small blocks
for k in range(3):
    x=5.04+k*.65
    box(x,5.20,.52,.39,fill=['#E2F0F5','#D4EAE9','#E2F0F5'][k],edge=TEAL,lw=.75,r=.05)
    for j in range(3): line((x+.10,5.29+j*.08),(x+.42,5.29+j*.08),TEAL,.9)
arrow((7.10,5.39),(7.69,5.39),TEAL,1.5,ms=10)
box(7.73,5.19,.47,.41,fill='#E5F3EE',edge=GREEN,lw=1,r=.07)
text(7.965,5.40,'GPU',7.5,GREEN,'bold',ha='center')
# float conversion
text(9.02,5.49,'64',17,PURPLE,'bold')
arrow((9.65,5.48),(10.30,5.48),PURPLE,1.5,ms=10)
text(10.49,5.49,'32',17,TEAL,'bold')
box(11.20,5.24,1.13,.39,fill='#E8F5F0',edge='#D6EDE3',r=.14)
text(11.765,5.435,'✓  CHECKED',7.5,GREEN,'bold',ha='center')

# Central panel
box(.56,1.13,12.20,3.43,fill=WHITE,edge=BORDER,lw=.95,r=.15)
text(.82,4.24,'THE CORE IDEA',10.3,INK,'bold')
text(.82,3.94,'One spatial batch enters the GPU with its entire hourly time axis intact.',9.5,MUTED)

# data cube, axonometric using physical inch coordinates
origin=np.array([1.25,2.03]); ex=np.array([2.24,0.0]); ey=np.array([.68,.54]); ez=np.array([0.,1.17])
def poly(points,fill,edge='#FFFFFF',lw=.8,alpha=1):
    ax.add_patch(Polygon(points,closed=True,facecolor=fill,edgecolor=edge,lw=lw,alpha=alpha,joinstyle='round'))
A=origin; B=A+ex; C=B+ey; D=A+ey; E=A+ez; F=B+ez; G=C+ez; I=D+ez
poly([A,B,F,E],'#DDECF5',edge=BLUE,lw=1.25)
poly([B,C,G,F],'#9FCBD0',edge=TEAL,lw=1.1)
poly([E,F,G,I],'#E8F1F7',edge=BLUE,lw=1.1)
# lines across full time, never break the time axis
for j in range(1,5):
    t=j/5
    line(A+ez*t,B+ez*t,BLUE,.95,.75)
    line(B+ez*t,C+ez*t,TEAL,.75,.6)
for j in range(1,4):
    t=j/4
    line(A+ex*t,E+ex*t,BLUE,.7,.45)
for j in range(1,3):
    t=j/3
    line(E+ey*t,F+ey*t,BLUE,.7,.45)
text(2.66,1.75,'SPACE / POINTS',7.9,BLUE,'bold',ha='center')
arrow((1.01,2.03),(1.01,3.20),ORANGE,1.6,ms=11)
text(.90,2.65,'TIME',8.4,ORANGE,'bold',rotation=90,ha='center')
text(2.80,3.67,'COMPLETE TIME SERIES',8.1,TEAL,'bold',ha='center')

# separated spatial slabs and CPU/GPU memory depiction
arrow((4.35,2.80),(5.00,2.80),TEAL,2.0,ms=14)
box(5.05,1.70,2.03,1.90,fill='#F1F6FA',edge='#C8DCE8',lw=1.1,r=.13)
text(6.065,3.34,'CPU / RAM',9.8,BLUE,'bold',ha='center')
for j in range(3):
    yy=2.94-j*.36
    box(5.31,yy-.13,1.47,.25,fill=['#D9EAF4','#E2F1F0','#D9EAF4'][j],edge='none',r=.035)
    text(6.045,yy,f'Batch {j+1:02d}  ·  all t',8.0,INK,ha='center')
text(6.065,1.93,'Spatial batches',7.8,MUTED,ha='center')

arrow((7.12,2.70),(7.94,2.70),TEAL,2.0,ms=15)
arrow((7.94,2.36),(7.12,2.36),MUTED,1.0,ms=9)
text(7.52,3.04,'H2D',7.4,TEAL,'bold',ha='center')
text(7.52,2.11,'results',7.0,MUTED,ha='center')

box(8.04,1.70,2.13,1.90,fill='#E9F5F1',edge='#B8DCD2',lw=1.1,r=.13)
text(9.105,3.34,'GPU / VRAM',9.8,GREEN,'bold',ha='center')
# time series in GPU
xx=np.linspace(8.30,9.91,180)
yy=2.60+.16*np.sin(np.linspace(0,7*np.pi,180))+.12*np.sin(np.linspace(0,2*np.pi,180))
ax.plot(xx,yy,color=TEAL,lw=2.0)
line((8.30,2.51),(9.91,2.51),ORANGE,1.0,ls='--')
text(9.10,2.03,'Compute across all hours',7.8,MUTED,ha='center')

arrow((10.27,2.67),(10.91,2.67),TEAL,1.8,ms=13)
box(11.00,1.91,1.43,1.47,fill='#F4F1F9',edge='#D9D2EB',lw=.95,r=.12)
text(11.715,3.12,'REPEAT',10,PURPLE,'bold',ha='center')
text(11.715,2.66,'Batch 01',8.6,INK,ha='center')
text(11.715,2.38,'Batch 02',8.6,INK,ha='center')
text(11.715,2.10,'Batch 03 ···',8.6,INK,ha='center')

# Footline: concise and technically precise
pill(.58,.57,2.80,'CHUNK BY SPACE, NOT TIME',fill='#E5F1F5',color=BLUE)
pill(3.54,.57,2.64,'TIME AXIS NEVER SPLIT',fill='#E6F3F1',color=TEAL)
pill(6.36,.57,2.89,'FLOAT32  ·  VALIDATED',fill='#F0ECF8',color=PURPLE)
text(12.72,.70,'CUDAHeat',9.4,INK,'bold',ha='right')
text(12.72,.42,'Full history per point  /  bounded VRAM',7.6,MUTED,ha='right')

fig.savefig(OUT,dpi=DPI,bbox_inches=None,pad_inches=0)
#plt.close(fig)
print(f'Saved {OUT}  (6667 × 3750 px at {DPI} DPI)')