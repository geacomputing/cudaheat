#!/usr/bin/env python3
"""Read-only NetCDF I/O investigation, tailored to complete-time spatial batches.

Requires numpy, netCDF4. Optional psutil for RSS reporting.
No input modifications, no cache dropping, no output NetCDF conversion.
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


def serial(x):
    if isinstance(x, np.generic): return serial(x.item())
    if isinstance(x, np.ndarray): return [serial(z) for z in x.tolist()]
    if isinstance(x, (list, tuple)): return [serial(z) for z in x]
    if isinstance(x, dict): return {str(k): serial(v) for k,v in x.items()}
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, (str,int,float,bool)) or x is None: return x
    return str(x)


def getvar(ds, name):
    node = ds
    for part in name.strip('/').split('/')[:-1]: node = node.groups[part]
    return node.variables[name.strip('/').split('/')[-1]]


def rss_mib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError: return None


def counts(sel, shape):
    return [len(range(*s.indices(n))) for s,n in zip(sel,shape)]


def estimate_chunk_access(sel, shape, chunks, dtype_bytes):
    if not chunks: return None
    total = 1
    for s,n,c in zip(sel,shape,chunks):
        r = range(*s.indices(n))
        if not len(r): return {'chunks':0,'chunk_uncompressed_MiB':0}
        # Exact for positive-stride slices, including sparse samples.
        if s.step is None or s.step == 1:
            hits = r[-1]//c-r[0]//c+1
        else:
            # Number of chunks touched by this 1D slice without enumerating time values.
            hits = len({i//c for i in r})
        total *= hits
    return {'chunks':int(total),'chunk_uncompressed_MiB':round(total*math.prod(chunks)*dtype_bytes/2**20,3)}


def timed(var, selector, mode, scale, offset, fill):
    var.set_auto_maskandscale(mode=='cf')
    var.set_auto_chartostring(False)
    start=time.perf_counter()
    data=var[selector]
    if mode=='manual32':
        a=np.asarray(data)
        mask=(a==fill) if fill is not None else None
        out=a.astype(np.float32)
        out *= np.float32(scale)
        out += np.float32(offset)
        if mask is not None: out[mask]=np.nan
        data=out
    elapsed=time.perf_counter()-start
    a=np.asarray(data)
    return elapsed,int(a.nbytes),str(a.dtype)


def bench(var, selector, label, mode, repeats, budget, scale, offset, fill, chunks):
    sizes=counts(selector,var.shape)
    raw_bytes=math.prod(sizes)*np.dtype(var.dtype).itemsize
    estimated_peak=raw_bytes*(5 if mode=='cf' else 3 if mode=='manual32' else 1)
    if estimated_peak>budget:
        return {'skipped':True,'reason':'Estimated allocation exceeds max-read-mib','estimated_peak_MiB':round(estimated_peak/2**20,1),'shape':sizes}
    times=[]; before=rss_mib(); peak=before
    nbytes=0; outdtype=''
    for i in range(repeats):
        t,nbytes,outdtype=timed(var,selector,mode,scale,offset,fill)
        times.append(t)
        current=rss_mib()
        if current is not None: peak=max(peak or current,current)
    median=statistics.median(times)
    result={'seconds':[round(t,6) for t in times], 'median_seconds':round(median,6),
            'first_seconds':round(times[0],6), 'output_bytes':nbytes,'output_dtype':outdtype,
            'output_MiB_per_s':round(nbytes/2**20/max(median,1e-9),2),
            'shape':sizes,'mode':mode,'rss_before_MiB':round(before,2) if before is not None else None,
            'rss_max_observed_MiB':round(peak,2) if peak is not None else None,
            'chunk_access_estimate':estimate_chunk_access(selector,var.shape,chunks,np.dtype(var.dtype).itemsize)}
    print(f'{label:48s} {median:9.3f} s  {nbytes/2**20:9.1f} MiB',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('file',type=Path)
    p.add_argument('--variable',default='analysed_sst')
    p.add_argument('--points',type=int,default=2000,help='Target CUDAHeat-like spatial batch size')
    p.add_argument('--batch-sizes',default='64,128,256,512,1024,2000',help='Comma-separated spatial batch sizes')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-read-mib',type=float,default=2048,help='Approximate maximum temporary memory for a test, not process-wide hard limit')
    p.add_argument('--time-hours',type=int,default=24)
    p.add_argument('--full-sweep',action='store_true',help='Read entire domain in complete-time spatial batches (potentially slow)')
    p.add_argument('--sweep-points',type=int,default=2000)
    p.add_argument('--sweep-time-block',type=int,default=0,help='0=full time; otherwise time blocking (not CUDAHeat equivalent)')
    p.add_argument('--mask-sample-hours',type=int,default=0,help='Sample this many evenly spaced times to estimate permanent-land fraction (NOT definitive)')
    p.add_argument('--json',default='netcdf_io_extended_report.json')
    a=p.parse_args()
    if min(a.points,a.repeats,a.max_read_mib,a.time_hours,a.sweep_points)<=0: p.error('Numeric parameters must be positive')
    batches=sorted(set(int(v.strip()) for v in a.batch_sizes.split(',')))
    if not batches or min(batches)<=0:p.error('Invalid --batch-sizes')
    path=a.file.expanduser().resolve()
    if not path.is_file():p.error(f'File not found: {path}')
    out=Path(a.json).expanduser().resolve()
    if out==path: p.error('JSON output must not overwrite input')
    report={'schema_version':'2.0','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'netcdf4_c':__netcdf4libversion__,'platform':platform.platform()},
            'file':{'basename':path.name,'size_GiB':round(path.stat().st_size/2**30,3)},
            'settings':vars(a).copy(),'tests':{},'warnings':['Read timings are warm-cache influenced. No OS cache eviction is attempted.',
            'Chunk-access estimates represent intersected uncompressed chunks, not measured disk I/O or decompression time.',
            'The memory budget is estimated, not an enforced RSS cap.']}
    report['settings']['file']=str(path)
    with Dataset(path,'r') as ds:
        v=getvar(ds,a.variable)
        if len(v.shape)!=3:raise ValueError('Requires a 3D (time, latitude, longitude) variable')
        dims=list(v.dimensions)
        def identify(names):
            hits=[i for i,d in enumerate(dims) if d.lower() in names]
            if len(hits)!=1:raise ValueError(f'Cannot identify dimension from {dims}')
            return hits[0]
        ti=identify({'time','t'}); yi=identify({'latitude','lat','y'}); xi=identify({'longitude','lon','x'})
        nt,ny,nx=v.shape[ti],v.shape[yi],v.shape[xi]
        rawchunks=v.chunking(); chunks=rawchunks if isinstance(rawchunks,list) else None
        scale=float(getattr(v,'scale_factor',1.0)); offset=float(getattr(v,'add_offset',0.0))
        fill=getattr(v,'_FillValue',None)
        report['variable']={'dimensions':dims,'shape':list(v.shape),'dtype':str(v.dtype),
          'chunks':rawchunks,'filters':v.filters(),'attributes':{k:serial(v.getncattr(k)) for k in v.ncattrs()},
          'spatial_points':ny*nx,'uncompressed_GiB':round(math.prod(v.shape)*np.dtype(v.dtype).itemsize/2**30,3),
          'compression_ratio_vs_raw':round(math.prod(v.shape)*np.dtype(v.dtype).itemsize/path.stat().st_size,3)}
        budget=int(a.max_read_mib*2**20)
        def sel(t,y,x):
            s=[slice(None)]*3;s[ti]=t;s[yi]=y;s[xi]=x;return tuple(s)
        def test(label,s,mode='raw',rep=None):
            report['tests'][label]=bench(v,s,label,mode,rep or a.repeats,budget,scale,offset,fill,chunks)
        cy=ny//2;cx=nx//2
        test('single_point_full_raw',sel(slice(None),slice(cy,cy+1),slice(cx,cx+1)))
        test('single_point_full_cf',sel(slice(None),slice(cy,cy+1),slice(cx,cx+1)),'cf')
        for count in batches:
            # A square-ish tile and a row-spanning rectangle with roughly same point count.
            ty=min(ny,max(1,round(math.sqrt(count))))
            tx=min(nx,max(1,count//ty))
            ty=min(ny,max(1,count//tx))
            sy=min(ny-ty,max(0,cy-ty//2));sx=min(nx-tx,max(0,cx-tx//2))
            base=f'tile_{ty}x{tx}_{ty*tx}points'
            test(base+'_raw',sel(slice(None),slice(sy,sy+ty),slice(sx,sx+tx)))
            if count==min(batches,key=lambda z:abs(z-a.points)):
                test(base+'_cf',sel(slice(None),slice(sy,sy+ty),slice(sx,sx+tx)),'cf')
                test(base+'_manual32',sel(slice(None),slice(sy,sy+ty),slice(sx,sx+tx)),'manual32')
            # Row strip spanning full longitude, as an alternative to compact square tile.
            ry=min(ny,max(1,math.ceil(count/nx)));rx=nx if ry>1 else min(nx,count)
            sy=min(ny-ry,max(0,cy-ry//2))
            test(f'row_strip_{ry}x{rx}_raw',sel(slice(None),slice(sy,sy+ry),slice(0,rx)))
        test('short_time_full_grid_raw',sel(slice(0,min(nt,a.time_hours)),slice(None),slice(None)))
        if chunks:
            test('physical_chunk_raw',sel(slice(0,chunks[ti]),slice(0,chunks[yi]),slice(0,chunks[xi])))
        # Assess whether a spatial mask could reduce storage; a sampled mask is not definitive.
        if a.mask_sample_hours:
            sample_indices=np.unique(np.linspace(0,nt-1,min(nt,a.mask_sample_hours),dtype=np.int64))
            observed=np.zeros((ny,nx),dtype=bool)
            v.set_auto_maskandscale(False)
            for idx in sample_indices:
                frame=np.asarray(v[sel(slice(int(idx),int(idx)+1),slice(None),slice(None))]).squeeze(axis=ti)
                if fill is None: observed|=np.isfinite(frame)
                else: observed|=(frame!=fill)
            report['mask_sample']={'hours_sampled':len(sample_indices),'points_never_valid_in_sample':int((~observed).sum()),
              'fraction_never_valid_in_sample':round(float((~observed).mean()),5),
              'note':'Only a candidate land mask; intermittent missing observations can cause false land classification.'}
        if a.full_sweep:
            # Reproduce flattened spatial-batch order, using one or more row slices per batch.
            # Time is kept complete unless explicitly requested otherwise.
            sweep={'batch_points':a.sweep_points,'time_block':a.sweep_time_block or nt,'mode':'raw',
                   'strategy':'flat contiguous points, decomposed into contiguous row segments',
                   'batches':0,'reads':0,'output_GiB':0,'total_seconds':0,'per_batch_seconds':[]}
            v.set_auto_maskandscale(False)
            tblock=min(nt,a.sweep_time_block or nt)
            # Guard: each segment at most one row, but very long full-time row can be large.
            max_row_bytes=tblock*nx*np.dtype(v.dtype).itemsize
            if max_row_bytes>budget:
                sweep['skipped']=True;sweep['reason']='One row segment exceeds memory budget; use --sweep-time-block'
            else:
                start_sweep=time.perf_counter()
                for p0 in range(0,ny*nx,a.sweep_points):
                    p1=min(ny*nx,p0+a.sweep_points)
                    bt=time.perf_counter()
                    for tt in range(0,nt,tblock):
                        tsel=slice(tt,min(nt,tt+tblock))
                        pos=p0
                        while pos<p1:
                            row=pos//nx;col=pos%nx
                            end=min(p1,(row+1)*nx)
                            s=sel(tsel,slice(row,row+1),slice(col,col+end-pos))
                            dat=v[s]
                            sweep['output_GiB']+=np.asarray(dat).nbytes/2**30
                            sweep['reads']+=1
                            pos=end
                    sweep['batches']+=1
                    sweep['per_batch_seconds'].append(round(time.perf_counter()-bt,5))
                    if sweep['batches']%5==0:print(f'  sweep: {sweep["batches"]} batches, {p1}/{ny*nx} points',flush=True)
                sweep['total_seconds']=round(time.perf_counter()-start_sweep,3)
                sweep['output_GiB']=round(sweep['output_GiB'],3)
                sweep['throughput_MiB_per_s']=round(sweep['output_GiB']*1024/max(sweep['total_seconds'],1e-9),2)
            report['full_sweep']=sweep
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(serial(report),indent=2)+'\n')
    print(f'\nJSON report: {out}',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
