#!/usr/bin/env python3
"""Dragonera hourly SST: PowerPoint-sized time series and missing-data diagnostics."""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

DATA_DIR = Path('/home/user/Documents/PhD.cy/Dragonera/data')
OUTPUT_DIR = Path(__file__).resolve().parent
SST_FILE = DATA_DIR / '25341_52135_2820_WATER_TEMP_20070101122249_20260502112249.csv'
NULL_VALUE = -9999.9
DPI = 500


def load_sst(path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f'SST CSV not found: {path}')
    df = pd.read_csv(path, sep='\t', skiprows=1, header=0, encoding='latin1')
    df = df.dropna(axis=1, how='all')
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], format='%Y %m %d %H', errors='coerce')
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    candidates = [c for c in df.columns if 'canal' not in str(c).lower()]
    if not candidates:
        raise ValueError('No SST data column found.')
    s = pd.to_numeric(df[candidates[0]], errors='coerce')
    s = s.mask(np.isclose(s, NULL_VALUE, rtol=0, atol=0.001))
    s = s[~s.index.duplicated(keep='first')]
    if s.empty:
        raise ValueError('No valid timestamps in CSV.')
    # Explicit hourly axis ensures absent timestamps are counted as gaps.
    time = pd.date_range(s.index.min(), s.index.max(), freq='h')
    return s.reindex(time).rename('sst')


def gap_table(s: pd.Series) -> pd.DataFrame:
    missing = s.isna().to_numpy()
    transitions = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)  # exclusive
    gaps = pd.DataFrame({
        'start': s.index[starts],
        'end': s.index[ends - 1],
        'duration_hours': ends - starts,
    })
    gaps['duration_days'] = gaps['duration_hours'] / 24
    return gaps.sort_values('duration_hours', ascending=False).reset_index(drop=True)


def plot_gap(ax, s: pd.Series, row: pd.Series, number: int) -> None:
    start, end = pd.Timestamp(row['start']), pd.Timestamp(row['end'])
    duration = pd.Timedelta(hours=int(row['duration_hours']))
    # Show context on either side, with a minimum of 10 days and a maximum of 45.
    padding = pd.Timedelta(days=float(np.clip(duration / pd.Timedelta(days=1) * 0.35, 10, 45)))
    left, right = max(s.index.min(), start - padding), min(s.index.max(), end + padding)
    local = s.loc[left:right]
    ax.plot(local.index, local.values, color='#146DA0', lw=1.15)
    ax.axvspan(start - pd.Timedelta(minutes=30), end + pd.Timedelta(minutes=30),
               color='#D14D45', alpha=0.19, lw=0)
    ax.axvline(start, color='#C33E35', ls='--', lw=0.9)
    ax.axvline(end, color='#C33E35', ls='--', lw=0.9)
    ax.set_title(f'Gap {number}  |  {int(row["duration_hours"]):,} h ({row["duration_days"]:,.1f} days)',
                 loc='left', fontsize=12, weight='bold', pad=10)
    ax.text(0.02, 0.96, f'{start:%d %b %Y} – {end:%d %b %Y}',
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
    ax.set_xlim(left, right)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(alpha=0.2, lw=0.6)
    ax.set_ylabel('SST (°C)', fontsize=10)


def main() -> None:
    s = load_sst(SST_FILE)
    gaps = gap_table(s)
    gaps.to_csv(OUTPUT_DIR / 'dragonera_sst_gaps.csv', index=False)
    xr.Dataset({'sst': ('time', s.to_numpy(dtype=np.float32), {'units': 'degC'})},
               coords={'time': s.index},
               attrs={'title': 'Dragonera hourly sea surface temperature',
                      'missing_data_policy': 'Missing timestamps and source nulls represented as NaN',
                      'source': str(SST_FILE)}).to_netcdf(OUTPUT_DIR / 'dragonera_sst_hourly.nc')

    print(f'Time span: {s.index.min()} to {s.index.max()}')
    print(f'Hourly samples: {len(s):,}; valid: {s.notna().sum():,}; missing: {s.isna().sum():,} '
          f'({s.isna().mean():.1%})')
    print(f'Number of missing intervals: {len(gaps):,}')
    print(gaps.head(10).to_string(index=False))

    # Exactly 13⅓ × 7½ inches -> 4000 × 2250 pixels at 300 DPI.
    fig = plt.figure(figsize=(40 / 3, 7.5), dpi=DPI, facecolor='white')
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.28,
                            left=0.06, right=0.98, bottom=0.09, top=0.89)
    top = outer[0].subgridspec(1, 3, wspace=0.28)
    axes = [fig.add_subplot(top[0, i]) for i in range(3)]
    bottom = fig.add_subplot(outer[1])

    fig.suptitle('DRAGONERA  |  Hourly sea surface temperature and data gaps',
                 x=0.06, y=0.965, ha='left', fontsize=15, weight='bold')
    # fig.text(0.06, 0.912,
    #          f'{s.index.min():%Y}–{s.index.max():%Y}  •  '
    #          f'{len(s):,} expected hourly records  •  '
    #          f'{s.isna().mean():.1%} missing  •  ',
    #          #f'{len(gaps):,} missing intervals',
    #          ha='left', fontsize=12, color='#555555')

    for i, ax in enumerate(axes):
        if i < len(gaps):
            plot_gap(ax, s, gaps.iloc[i], i + 1)
        else:
            ax.text(0.5, 0.5, 'No additional gap', ha='center', va='center',
                    transform=ax.transAxes, fontsize=13)
            ax.set_axis_off()

    bottom.plot(s.index, s.values, color='#146DA0', lw=0.55, rasterized=True)
    bottom.set_title('Complete hourly SST record', loc='left', fontsize=15,
                     weight='bold', pad=12)
    bottom.set_xlim(s.index.min(), s.index.max())
    bottom.set_ylabel('SST (°C)', fontsize=12)
    bottom.set_xlabel('Year', fontsize=12)
    bottom.xaxis.set_major_locator(mdates.YearLocator(2))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    bottom.grid(alpha=0.23, lw=0.7)
    bottom.tick_params(labelsize=10)
    fig.text(0.98, 0.025, 'Missing observations are not interpolated.',
             ha='right', fontsize=10, color='#666666')
    
    
    
    # ============================================================
# Methodological annotation — lower panel
# ============================================================

    bottom.text(
        0.70, 0.99,
        "Treatment of missing observations remains\n"
        "an open methodological issue.",
        transform=bottom.transAxes,
        fontsize=15,
#        fontweight="bold",
        fontstyle="italic",
        color="red",
        rotation=0,
        ha="center",
        va="center",
        alpha=0.85,
        zorder=20, 
        bbox=dict(
        boxstyle="square,pad=0.5",
        facecolor="white",
        edgecolor="gray",
        linewidth=1.8,
        alpha=0.5
    )
    )
        
    
    

    outfile = OUTPUT_DIR / 'dragonera_sst_gaps_16x9.png'
    fig.savefig(outfile, dpi=DPI, bbox_inches=None, pad_inches=0, transparent=False)
    plt.show()
    print(f'\nSaved: {outfile} (4000 × 2250 pixels)')


if __name__ == '__main__':
    plt.close("all")
    main()