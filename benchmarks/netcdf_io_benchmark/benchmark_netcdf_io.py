#!/usr/bin/env python3
"""Read-only, standalone NetCDF I/O diagnostic for time x lat x lon arrays.

Requires: pip install netCDF4 numpy
No CUDAHeat dependency. Does not write or rechunk the input file.
"""
import argparse
import datetime as dt
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, __netcdf4libversion__


def clean(x):
    if isinstance(x, (np.integer, np.floating)):
        return clean(x.item())
    if isinstance(x, (np.ndarray, list, tuple)):
        return [clean(y) for y in x]
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def find_variables(group, prefix=''):
    out = {}
    for name, var in group.variables.items():
        out[prefix + name] = var
    for name, sub in group.groups.items():
        out.update(find_variables(sub, prefix + name + '/'))
    return out


def pick_dims(var, args):
    dims = list(var.dimensions)
    if len(dims) != 3:
        raise ValueError('Benchmark expects a 3D variable (time, latitude, longitude). Use --time-dim/--lat-dim/--lon-dim to identify dimensions.')
    def choose(explicit, candidates):
        if explicit:
            if explicit not in dims:
                raise ValueError(f'Dimension {explicit!r} not in {dims}')
            return explicit
        matches = [d for d in dims if d.lower() in candidates]
        if len(matches) != 1:
            raise ValueError(f'Cannot identify dimensions from {dims}. Specify --time-dim, --lat-dim, --lon-dim.')
        return matches[0]
    t = choose(args.time_dim, {'time', 't'})
    y = choose(args.lat_dim, {'lat', 'latitude', 'y'})
    x = choose(args.lon_dim, {'lon', 'longitude', 'x'})
    if len({t,y,x}) != 3:
        raise ValueError('Time, latitude and longitude dimensions must differ')
    return t,y,x


def slices_for(dims, mapping):
    return tuple(mapping[d] for d in dims)


def mem_bytes(shape, dtype):
    return int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)


def run_read(var, selectors, decoded):
    var.set_auto_maskandscale(decoded)
    var.set_auto_chartostring(False)
    start = time.perf_counter()
    data = var[selectors]
    elapsed = time.perf_counter() - start
    # Force materialization / minimal inspection; netCDF4 indexing already eager.
    arr = np.asarray(data)
    return elapsed, int(arr.nbytes), str(arr.dtype)


def measure(var, selectors, repeats, decoded):
    trials = []
    for _ in range(repeats):
        seconds, nbytes, dtype = run_read(var, selectors, decoded)
        trials.append(seconds)
    return {
        'seconds': [round(s, 6) for s in trials],
        'median_seconds': round(statistics.median(trials), 6),
        'first_seconds': round(trials[0], 6),
        'last_seconds': round(trials[-1], 6),
        'output_bytes': nbytes,
        'output_dtype': dtype,
        'effective_output_MiB_per_s': round((nbytes / 1048576) / max(statistics.median(trials), 1e-9), 2),
        'note': 'Repeated reads may benefit from OS/HDF5 caches; not independent cold-cache trials.'
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('file', help='Input NetCDF file (read-only)')
    p.add_argument('--variable', default='analysed_sst')
    p.add_argument('--time-dim')
    p.add_argument('--lat-dim')
    p.add_argument('--lon-dim')
    p.add_argument('--points', type=int, default=128, help='Target neighbouring spatial points in a full-time read')
    p.add_argument('--time-hours', type=int, default=24, help='Short time window for comparative spatial reads')
    p.add_argument('--max-read-mib', type=float, default=256, help='Hard limit on any one raw data read, estimated from dtype')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--no-benchmark', action='store_true', help='Only inspect metadata')
    p.add_argument('--json', default='netcdf_io_report.json')
    args = p.parse_args()
    if args.repeats < 1 or args.points < 1 or args.max_read_mib <= 0:
        p.error('repeats, points and max-read-mib must be positive')
    path = Path(args.file).expanduser().resolve()
    if not path.is_file():
        p.error(f'File not found: {path}')
    report = {
        'schema_version': '1.0',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'environment': {'python': sys.version.split()[0], 'numpy': np.__version__, 'netcdf4_c': __netcdf4libversion__, 'platform': platform.platform()},
        'file': {'basename': path.name, 'size_bytes': path.stat().st_size, 'size_GiB': round(path.stat().st_size / 2**30, 3)},
        'settings': {'points': args.points, 'time_hours': args.time_hours, 'max_read_mib': args.max_read_mib, 'repeats': args.repeats},
        'tests': {}, 'warnings': [],
    }
    with Dataset(str(path), 'r') as ds:
        variables = find_variables(ds)
        if args.variable not in variables:
            p.error(f'Variable {args.variable!r} not found. Available: {list(variables)}')
        var = variables[args.variable]
        td, yd, xd = pick_dims(var, args)
        sizes = dict(zip(var.dimensions, var.shape))
        nt, ny, nx = sizes[td], sizes[yd], sizes[xd]
        chunks = var.chunking()
        chunk_map = dict(zip(var.dimensions, chunks)) if isinstance(chunks, list) else None
        filters = var.filters() if hasattr(var, 'filters') else None
        attrs = {}
        for name in var.ncattrs():
            value = var.getncattr(name)
            if name in {'scale_factor', 'add_offset', '_FillValue', 'missing_value', 'units', 'standard_name', 'long_name', 'valid_min', 'valid_max', 'valid_range'}:
                attrs[name] = clean(value)
        dtype = np.dtype(var.dtype)
        chunk_shape = [int(v) for v in chunks] if isinstance(chunks, list) else None
        report['variable'] = {
            'name': args.variable, 'dimensions': list(var.dimensions), 'shape': list(var.shape),
            'dimension_sizes': sizes, 'dtype_on_disk': str(dtype), 'dtype_bytes': dtype.itemsize,
            'uncompressed_GiB': round(mem_bytes(var.shape, dtype) / 2**30, 3),
            'physical_chunking': clean(chunks), 'physical_chunks_by_dim': chunk_map,
            'chunk_uncompressed_MiB': round(mem_bytes(chunk_shape, dtype)/2**20, 3) if chunk_shape else None,
            'total_physical_chunks': int(math.prod(math.ceil(n / c) for n,c in zip(var.shape, chunks))) if chunk_shape else None,
            'filters': clean(filters), 'attributes': attrs,
            'time_length': nt, 'spatial_points': ny*nx,
            'time_chunks_per_complete_series': math.ceil(nt / chunk_map[td]) if chunk_map else None,
        }
        if chunk_map:
            report['variable']['one_series_chunk_amplification_estimate'] = round(chunk_map[yd]*chunk_map[xd], 2)
            report['variable']['one_series_estimated_uncompressed_chunk_MiB_total'] = round(math.ceil(nt/chunk_map[td])*mem_bytes(chunk_shape, dtype)/2**20, 2)
            report['warnings'].append('Chunk amplification estimate is approximate: chunk edges, compression, cache reuse and internal HDF5 access affect actual bytes read.')
        if filters and any(filters.get(k) for k in ('zlib', 'szip', 'zstd', 'bzip2', 'blosc')):
            report['warnings'].append('Variable is compressed: partial selections can require decompressing complete overlapping chunks.')
        if not args.no_benchmark:
            budget = int(args.max_read_mib * 2**20)
            def add_test(name, tsel, ysel, xsel, decoded=False):
                lengths = [len(range(*s.indices(sizes[d]))) if isinstance(s, slice) else 1 for d,s in zip(var.dimensions, slices_for(var.dimensions, {td:tsel, yd:ysel, xd:xsel}))]
                requested = mem_bytes(lengths, dtype)
                if requested > budget:
                    report['tests'][name] = {'skipped': True, 'reason': f'Estimated raw output {requested/2**20:.1f} MiB exceeds --max-read-mib {args.max_read_mib}'}
                    return
                report['tests'][name] = measure(var, slices_for(var.dimensions, {td:tsel, yd:ysel, xd:xsel}), args.repeats, decoded)
                report['tests'][name].update({'time_count': len(range(*tsel.indices(nt))), 'latitude_count': len(range(*ysel.indices(ny))), 'longitude_count': len(range(*xsel.indices(nx))), 'decoded': decoded})
                print(f'  {name:32} {report["tests"][name]["median_seconds"]:.3f} s  ({report["tests"][name]["effective_output_MiB_per_s"]:.1f} MiB/s)', flush=True)
            # All reads are bounded. Full-time test sizes are automatically limited by budget.
            safe_full_points = max(1, budget // max(nt*dtype.itemsize*2, 1)) # 2x margin for CF decode
            full_points = min(args.points, nx, safe_full_points)
            # Prefer an interior row/tile, to avoid atypical chunk boundaries at origin.
            y0 = ny // 2
            x0 = max(0, min(nx - full_points, nx//2 - full_points//2))
            tshort = min(nt, max(1,args.time_hours))
            print('Running bounded NetCDF read benchmarks (warm-cache timings may differ from cold disk reads):', flush=True)
            add_test('single_point_full_time_raw', slice(None), slice(y0,y0+1), slice(x0,x0+1))
            add_test('contiguous_points_full_time_raw', slice(None), slice(y0,y0+1), slice(x0,x0+full_points))
            add_test('contiguous_points_full_time_decoded', slice(None), slice(y0,y0+1), slice(x0,x0+full_points), decoded=True)
            add_test('contiguous_points_short_time_raw', slice(0,tshort), slice(y0,y0+1), slice(x0,x0+full_points))
            # Compare same number of points but spaced apart in longitude.
            step = max(1, nx // max(1,full_points))
            spaced_count = min(full_points, math.ceil((nx-x0)/step))
            add_test('strided_points_full_time_raw', slice(None), slice(y0,y0+1), slice(x0, min(nx,x0+spaced_count*step), step))
            # Read a modest 2D spatial tile, to compare with single-row contiguous layout.
            tile_y = min(ny, max(1, math.isqrt(full_points)))
            tile_x = min(nx, max(1, full_points//tile_y))
            tile_y = min(tile_y, max(1, safe_full_points//tile_x))
            yy = min(max(0,ny//2-tile_y//2), ny-tile_y)
            xx = min(max(0,nx//2-tile_x//2), nx-tile_x)
            add_test('rectangular_tile_full_time_raw', slice(None), slice(yy,yy+tile_y), slice(xx,xx+tile_x))
            # Explicit short-time large tile: read an on-disk chunk-shaped tile if possible.
            if chunk_map:
                cy = min(ny, chunk_map[yd]); cx = min(nx, chunk_map[xd]); ct = min(nt, chunk_map[td])
                add_test('one_physical_chunk_raw', slice(0,ct), slice(0,cy), slice(0,cx))
        else:
            report['warnings'].append('Benchmarks were disabled; no timing evidence available.')
    # Save report outside the opened file scope.
    output = Path(args.json).expanduser().resolve()
    if output == path:
        raise ValueError('JSON output cannot overwrite input NetCDF')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean(report), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'\nReport saved: {output}')
    print(f'File: {report["file"]["size_GiB"]} GiB | Variable: {report["variable"]["name"]} | Chunks: {chunks} | Compression: {filters}')
    print('Share the JSON report for analysis. This tool does not modify the NetCDF file.')

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
