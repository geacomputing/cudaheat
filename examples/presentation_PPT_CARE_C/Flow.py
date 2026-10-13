
    
  
#!/usr/bin/env python3
"""PowerPoint-ready, six-stage MHW research workflow (16:9, 6667x3750)."""
from pathlib import Path
import base64
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle, Polygon, Arc
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

OUT = Path(__file__).resolve().parent / 'mhw_research_workflow_500dpi.png'
DPI = 500
# Official GitHub Octocat mark, embedded for a self-contained script.
GITHUB_PNG_BASE64 = "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"
BG, INK, MUTED = '#F7F9FB', '#17283B', '#617287'
BLUE, TEAL, ORANGE = '#2475B6', '#087F8C', '#E36B45'
GRID, WHITE = '#E1E8EE', '#FFFFFF'
COLORS = [BLUE, TEAL, '#5A7DA8', '#4E96A6', '#8071AC', ORANGE]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig = plt.figure(figsize=(6667/DPI, 7.5), dpi=DPI, facecolor=BG)
ASPECT = fig.get_figwidth() / fig.get_figheight()

def round_marker(x, y, radius, **kwargs):
    """Physically circular marker on a widescreen, normalized-coordinate axis."""
    return Ellipse((x, y), 2 * radius / ASPECT, 2 * radius, **kwargs)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

fig.text(.055, .925, 'FROM HOURLY SST TO MHW MECHANISMS', color=INK,
         fontsize=18, weight='bold', ha='left')
fig.text(.055, .878, 'A six-stage research workflow  /  Eastern Mediterranean and Cyprus',
         color=MUTED, fontsize=9.5, ha='left')

# Six cards across one horizontal row; spacing reserved for connectors.
n = 6; left, right, gap = .055, .945, .018
w = (right-left-(n-1)*gap)/n
bottom, height = .270, .425

def line(x1,y1,x2,y2,c,lw=2,alpha=1):
    ax.add_line(Line2D([x1,x2],[y1,y2], color=c, lw=lw, alpha=alpha,
                       solid_capstyle='round', transform=ax.transAxes))

def icon(i,cx,cy,col):
    """Small custom vector pictograms, no external assets required."""
    if i == 0:  # daily -> hourly: two traces
        x = np.linspace(cx-.043,cx+.043,9)
        ys = cy+.014+np.array([0,.004,.008,.013,.01,.017,.016,.022,.02])
        ax.plot(x,ys,color=BLUE,lw=2.4,marker='o',ms=2.8,transform=ax.transAxes)
        xx = np.linspace(cx-.043,cx+.043,80)
        yy = cy-.026 + .014*np.sin(np.linspace(0,5*np.pi,80)) + .010*np.linspace(0,1,80)
        ax.plot(xx,yy,color=TEAL,lw=1.9,transform=ax.transAxes)
    elif i == 1:  # GPU / code brackets
        ax.text(cx,cy+.006,'⟨ / ⟩',ha='center',va='center',fontsize=27,
                weight='bold',color=col,transform=ax.transAxes)
        for dx in [-.025,-.012,.001,.014,.027]:
            ax.add_patch(Rectangle((cx+dx-.003,cy-.042),.006,.007,
                                   transform=ax.transAxes,fc=col,ec='none'))
    elif i == 2:  # benchmarking bars
        for j,h in enumerate([.022,.037,.055,.043]):
            ax.add_patch(Rectangle((cx-.040+j*.023,cy-.034),.013,h,
                                   transform=ax.transAxes,fc=col,alpha=.72+j*.08,ec='none'))
        line(cx-.046,cy-.035,cx+.052,cy-.035,MUTED,1.2)
    elif i == 3:  # incoming data stream
        for k in range(3):
            yy=cy+.026-k*.025
            line(cx-.047,yy,cx+.018,yy,col,2.2)
            ax.add_patch(Polygon([[cx+.018,yy+.007],[cx+.033,yy],[cx+.018,yy-.007]],
                                 closed=True,transform=ax.transAxes,fc=col,ec='none'))
        ax.add_patch(Rectangle((cx+.040,cy-.036),.013,.077,
                               transform=ax.transAxes,fc='none',ec=col,lw=1.6))
    elif i == 4:  # taxonomy branches
        line(cx,cy+.039,cx,cy+.006,col,2)
        line(cx-.037,cy+.006,cx+.037,cy+.006,col,2)
        for dx in [-.037,0,.037]:
            line(cx+dx,cy+.006,cx+dx,cy-.018,col,2)
            ax.add_patch(round_marker(cx+dx,cy-.025,.009,transform=ax.transAxes,
                                facecolor=col,edgecolor='none'))
        ax.add_patch(round_marker(cx,cy+.044,.011,transform=ax.transAxes,fc=WHITE,ec=col,lw=2))
    else:  # thermal cage: basin + atmospheric layer
        xx=np.linspace(cx-.047,cx+.047,80)
        ax.plot(xx,cy-.023+.007*np.sin(np.linspace(0,4*np.pi,80)),
                color=TEAL,lw=2,transform=ax.transAxes)
        ax.add_patch(Arc((cx,cy-.005),.088,.096,theta1=12,theta2=168,
                         color=ORANGE,lw=2.6,transform=ax.transAxes))
        for dx in [-.023,0,.023]:
            line(cx+dx,cy-.017,cx+dx,cy+.016,ORANGE,1.3,.75)
            ax.add_patch(Polygon([[cx+dx-.005,cy+.010],[cx+dx,cy+.020],
                                  [cx+dx+.005,cy+.010]],transform=ax.transAxes,
                                 fc=ORANGE,ec='none'))

# Milestone timeline: completed 01-02, active 03, planned 04-06.
timeline_y = .785
centers = [left + i*(w+gap) + w/2 for i in range(n)]
line(centers[0],timeline_y,centers[-1],timeline_y,GRID,3.8)
line(centers[0],timeline_y,centers[2],timeline_y,TEAL,3.8)
for i, x in enumerate(centers):
    done, active = i < 2, i == 2
    fc = TEAL if done else (WHITE if active else BG)
    ec = TEAL if (done or active) else '#BAC7D3'
    ax.add_patch(round_marker(x,timeline_y,.012 if active else .009,
                        transform=ax.transAxes,fc=fc,ec=ec,lw=2.0,zorder=20))
    if done:
        ax.text(x,timeline_y,'✓',transform=ax.transAxes,color=WHITE,
                fontsize=9,weight='bold',ha='center',va='center',zorder=21)
    elif active:
        ax.add_patch(round_marker(x,timeline_y,.0048,transform=ax.transAxes,
                            fc=ORANGE,ec='none',zorder=21))
ax.text(centers[2],timeline_y+.030,'CURRENTLY TESTING',transform=ax.transAxes,
        ha='center',va='bottom',fontsize=8,weight='bold',color=ORANGE)
ax.text(centers[0],timeline_y+.030,'COMPLETED',transform=ax.transAxes,
        ha='center',va='bottom',fontsize=8,weight='bold',color=TEAL)
ax.text(centers[4],timeline_y+.030,'UPCOMING',transform=ax.transAxes,
        ha='center',va='bottom',fontsize=8,weight='bold',color=MUTED)

stages = [
    ('01', 'METHODOLOGY', 'Daily → hourly', 'Define hourly MHW\ncriteria and lifecycle'),
    ('02', 'GPU SOFTWARE', 'CUDAHeat', 'Develop GPU-powered,\nversion-controlled code'),
    ('03', 'VALIDATION', 'Test & optimise', 'Benchmark accuracy,\nspeed and scalability'),
    ('04', 'OBSERVATIONS', 'Ingest hourly SST', 'Apply to real data\nfor the study area'),
    ('05', 'EVENT TAXONOMY', 'Classify MHWs', 'Categorise event\nphases and behaviour'),
    ('06', 'WRF COUPLING', 'Thermal cage', 'Investigate air–sea\nfeedbacks near Cyprus'),
]
for i,(num,tag,title,desc) in enumerate(stages):
    x=left+i*(w+gap); cx=x+w/2; color=COLORS[i]
    ax.add_patch(FancyBboxPatch((x,bottom),w,height,
                 boxstyle='round,pad=0.003,rounding_size=0.014',
                 transform=ax.transAxes,facecolor=WHITE,
                 edgecolor=(COLORS[i] if i < 2 else GRID),
                 lw=(1.15 if i < 2 else 0.75)))
    ax.add_patch(Rectangle((x+.003,bottom+height-.009),w-.006,.006,
                           transform=ax.transAxes,fc=color,ec='none'))
    ax.text(x+.014,bottom+height-.056,num,transform=ax.transAxes,
            fontsize=10,weight='bold',color=color,va='center')
    ax.text(cx,bottom+height-.113,tag,transform=ax.transAxes,
            fontsize=7.0,weight='bold',color=MUTED,ha='center',va='center')
    icon(i,cx,bottom+height-(.184 if i == 4 else .160),color)
    ax.text(cx,bottom+.153,title,transform=ax.transAxes,
            fontsize=9.5,weight='bold',color=INK,ha='center',va='center')
    ax.text(cx,bottom+.093,desc,transform=ax.transAxes,
            fontsize=7.2,color=MUTED,ha='center',va='center',linespacing=1.65)
    if i==1:
        # Prefer a local github.png, otherwise use the embedded official mark.
        icon_file = Path(__file__).resolve().parent / 'github.png'
        if icon_file.exists():
            github_image = Image.open(icon_file).convert('RGBA')
        else:
            github_image = Image.open(io.BytesIO(base64.b64decode(GITHUB_PNG_BASE64))).convert('RGBA')
        github_pixels = np.array(github_image)
        # Keep original alpha/silhouette; turn visible artwork solid black.
        github_pixels[..., :3] = 0
        github_image = Image.fromarray(github_pixels, 'RGBA')
        bx, by = cx-.026, bottom+.038
        # OffsetImage preserves the original image aspect ratio in DISPLAY pixels.
        # imshow(extent=..., transform=ax.transAxes) distorted it on 16:9 axes.
        github_artist = AnnotationBbox(
            OffsetImage(np.asarray(github_image), zoom=0.05, resample=True),
            (bx, by), xycoords=ax.transAxes, frameon=False,
            box_alignment=(0.5, 0.5), pad=0, zorder=12,
        )
        ax.add_artist(github_artist)
        ax.text(bx+.019, by, 'GitHub', transform=ax.transAxes,
                fontsize=7.5, weight='bold', color=INK, va='center', ha='left')
    if i < n-1:
        mid=x+w+gap/2
        line(mid-.006,bottom+.244,mid+.004,bottom+.244,MUTED,1.5,.65)
        ax.add_patch(Polygon([[mid+.004,bottom+.250],[mid+.010,bottom+.244],
                              [mid+.004,bottom+.238]],transform=ax.transAxes,
                             fc=MUTED,ec='none',alpha=.75))

# Compact footer: remove the long horizontal research-logic sentence.
fig.text(.055, .155, 'RESEARCH PROGRESS', fontsize=9, weight='bold', color=INK)
fig.text(.055, .113, '2 milestones completed   /   Stage 3 in progress   /   3 stages planned',
         fontsize=9, color=MUTED)
fig.text(.945, .113, 'Eastern Mediterranean  ·  Cyprus',
         fontsize=8.5, color=MUTED, ha='right')
fig.savefig(OUT,dpi=DPI,bbox_inches=None,pad_inches=0,facecolor=BG)
#plt.close(fig)
print(f'Saved: {OUT}  ({int(round(fig.get_figwidth()*DPI))} × {int(round(fig.get_figheight()*DPI))} pixels)')
