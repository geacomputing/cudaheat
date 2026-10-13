#!/usr/bin/env python3
"""Conceptual comparison of daily and hourly SST for marine heatwave analysis.

Produces a 16:9, PowerPoint-ready PNG alongside this Python file.
All signals are synthetic and intended for methodological illustration.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

OUTPUT = Path(__file__).resolve().parent / 'hourly_vs_daily_mhw.png'
DPI = 500

# Deterministic synthetic hourly signal: broad warming pulse, day/night cycle,
# and a transient cool interruption.
time = pd.date_range('2025-07-01', periods=24 * 19, freq='h')
t = np.arange(len(time), dtype=float) / 24.0
warming = 2.75 / (1 + np.exp(-(t - 4.7) * 1.4))
cooling = 2.75 / (1 + np.exp(-(t - 14.1) * 1.2))
baseline = 27.0 + 0.12 * np.sin(2 * np.pi * t / 16)
heat = warming - cooling
cycle = 0.47 * np.sin(2 * np.pi * (t - 0.36)) * (0.7 + 0.3 * np.clip(heat / 2.75, 0, 1))
interruption = -0.85 * np.exp(-0.5 * ((t - 10.35) / 0.24) ** 2)
sst = baseline + heat + cycle + interruption
threshold = 28.75
hourly = pd.Series(sst, index=time)
daily = hourly.resample('D').mean()

# Clean presentation design.
BG = '#F7F9FB'
INK = '#17283B'
MUTED = '#617287'
BLUE = '#2475B6'
TEAL = '#087F8C'
ORANGE = '#E36B45'
GRID = '#E1E8EE'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.edgecolor': GRID,
    'axes.labelcolor': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'savefig.facecolor': BG,
})

fig = plt.figure(figsize=(40 / 3, 7.5), dpi=DPI, facecolor=BG)
fig.text(0.055, 0.935, 'ONE EVENT  /  TWO TEMPORAL PERSPECTIVES',
         fontsize=22, fontweight='bold', color=INK, ha='left')
fig.text(0.055, 0.882,
         'Daily averages describe the broad event; hourly observations resolve its evolution.',
         fontsize=13, color=MUTED, ha='left')

# Subtle central divider, two equally sized panels.
fig.add_artist(plt.Line2D([0.5, 0.5], [0.16, 0.83], transform=fig.transFigure,
                          color=GRID, lw=1.5))
ax_daily = fig.add_axes([0.065, 0.36, 0.405, 0.37])
ax_hourly = fig.add_axes([0.535, 0.36, 0.405, 0.37])

for ax in (ax_daily, ax_hourly):
    ax.set_facecolor('white')
    ax.set_ylim(26.3, 31.2)
    ax.set_xlim(time[0], time[-1])
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(GRID)
    if ax==ax_daily:
        ax.set_ylabel('SST (°C)', fontsize=11, color=MUTED)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(labelsize=10, length=0, pad=8)
    ax.axhline(threshold, color=ORANGE, ls=(0, (5, 4)), lw=1.8, zorder=2)
    ax.text(0.99, 0.51, 'Illustrative threshold', transform=ax.transAxes,
            ha='right', va='bottom', color=ORANGE, fontsize=9)

# Daily: each marker is one 24-hour mean.
ax_daily.plot(daily.index + pd.Timedelta(hours=12), daily.values,
              color=BLUE, lw=2.8, marker='o', ms=5, markerfacecolor='white',
              markeredgewidth=1.8, zorder=5)
ax_daily.fill_between(daily.index + pd.Timedelta(hours=12), daily.values,
                      threshold, where=daily.values > threshold,
                      color=BLUE, alpha=0.11)

# Hourly: all values; daily mean as a muted reference.
ax_hourly.plot(time, sst, color=TEAL, lw=1.25, alpha=0.98, zorder=5)
ax_hourly.plot(daily.index + pd.Timedelta(hours=12), daily.values,
               color=BLUE, lw=2, alpha=0.55, ls='--', zorder=4)
ax_hourly.fill_between(time, sst, threshold, where=sst > threshold,
                       color=TEAL, alpha=0.11, zorder=3)

# Mark the transient interruption on the hourly signal.
ax_hourly.annotate('Short cooling interruption',
                   xy=(time[int(10.35 * 24)], sst[int(10.35 * 24)]),
                   xytext=(0.55, 0.92), textcoords='axes fraction',
                   fontsize=10, color=INK, ha='center',
                   arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.2),
                   bbox=dict(facecolor='white', edgecolor='none', pad=2))

fig.text(0.065, 0.793, '01   DAILY SST', fontsize=17, weight='bold', color=BLUE)
fig.text(0.535, 0.793, '02   HOURLY SST', fontsize=17, weight='bold', color=TEAL)
fig.text(0.065, 0.755, 'One mean value per day', fontsize=11, color=MUTED)
fig.text(0.535, 0.755, '24 observations per day', fontsize=11, color=MUTED)

# Simple conceptual lifecycle bands below the graphs.
fig.text(0.065, 0.282, 'WHAT DAILY DATA REVEAL', fontsize=11, weight='bold', color=INK)
fig.text(0.535, 0.282, 'WHAT HOURLY DATA CAN REVEAL', fontsize=11, weight='bold', color=INK)
fig.text(0.065, 0.230,
         'Broad onset  •  persistence  •  decay', fontsize=13, color=BLUE)
fig.text(0.535, 0.230,
         'Diurnal heating  •  nocturnal retention  •  TAXONOMY',
         fontsize=12, color=TEAL)
fig.text(0.065, 0.178,
         'Sub-daily threshold crossings are averaged out.', fontsize=11, color=MUTED)
fig.text(0.535, 0.178,
         'Short excursions and cooling episodes remain visible.', fontsize=11, color=MUTED)

fig.text(0.055, 0.065,
         'CONCEPTUAL EXAMPLE  ·  Synthetic SST; fixed illustrative threshold. '
         'MHW event classification requires explicit duration and gap criteria.',
         fontsize=10.5, color=MUTED)

fig.savefig(OUTPUT, dpi=DPI, bbox_inches=None, pad_inches=0)
plt.close(fig)
print(f'Saved: {OUTPUT} (4000 × 2250 pixels)')