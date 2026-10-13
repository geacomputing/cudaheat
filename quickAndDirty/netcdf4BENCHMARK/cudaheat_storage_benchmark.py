#!/usr/bin/env python3
"""Standalone CUDAHeat NetCDF rechunking and simulated GPU-batch I/O benchmark.

pip install numpy netCDF4
Example:
 python cudaheat_storage_benchmark.py --input sst_8_years.nc --output-dir storage_test --chunks 4,8,16,24,32,48,64,96,128,192,256,512 --batch-points 64,128,256,512,1024,2048 --repeats 3

The benchmark reads complete time series, never splits time in GPU batches.
Only one optimized file is kept at a time by default; input is never modified.
"""
import argparse
import json
import math
import time
import warnings
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

def choose_dimension(dims, candidates, override):
    if override:
        if override not in dims:
            raise ValueError(f"{override} not among {dims}")
        return override
    found = [d for d in dims if d.lower() in candidates]
    if len(found) != 1:
        raise ValueError(f"Cannot resolve dimension from {dims}; specify dimension flags")
    return found[0]

def setup(src, name, args):
    v = src.variables[name]
    if v.ndim != 3:
        raise ValueError("Expected 3D SST variable")
    dims = v.dimensions
    t = choose_dimension(dims, {"time","t"}, args.time_dim)
    y = choose_dimension(dims, {"lat","latitude","y"}, args.lat_dim)
    x = choose_dimension(dims, {"lon","longitude","x"}, args.lon_dim)
    if len({t,y,x}) != 3:
        raise ValueError("Dimension names must be distinct")
    if tuple(dims) != (t,y,x):
        raise ValueError("This version expects variable order (time, latitude, longitude)")
    return v, (len(src.dimensions[t]),len(src.dimensions[y]),len(src.dimensions[x])), (t,y,x)

def raw(var, key):
    var.set_auto_maskandscale(False)
    var.set_auto_mask(False)
    return np.asarray(var[key])

def invalid_mask(var, data):
    # A point is permanently invalid if every sample is masked/fill/missing.
    # Do not treat isolated gaps as land.
    valid = np.ones(data.shape, dtype=bool)
    for attr in ("_FillValue", "missing_value"):
        if attr in var.ncattrs():
            for value in np.asarray(var.getncattr(attr)).reshape(-1):
                valid &= data != value
    if np.issubdtype(data.dtype, np.floating):
        valid &= np.isfinite(data)
    return ~valid

def scan_points(v, shape, scan_hours):
    T,Y,X = shape
    any_valid = np.zeros((Y,X), dtype=bool)
    for a in range(0,T,scan_hours):
        block = raw(v, (slice(a,min(T,a+scan_hours)),slice(None),slice(None)))
        any_valid |= np.any(~invalid_mask(v,block),axis=0)
    flat = np.flatnonzero(any_valid.ravel())
    if not len(flat):
        raise ValueError("No valid points found")
    return flat, int((~any_valid).sum())

def copy_coord(src, dst, dim):
    if dim in src.variables and src.variables[dim].ndim == 1:
        original=src.variables[dim]
        new=dst.createVariable(dim,original.dtype,(dim,))
        for attr in original.ncattrs():
            if attr != "_FillValue":
                new.setncattr(attr,original.getncattr(attr))
        new[:]=raw(original,slice(None))

def create_optimized(src, v, shape, dims, point_ids, out_path, chunk_points, compression, max_chunk_mib):
    T,Y,X=shape
    chunk_mib=T*chunk_points*v.dtype.itemsize/2**20
    if chunk_mib > max_chunk_mib:
        raise ValueError(f"Chunk (time={T}, points={chunk_points}) is {chunk_mib:.1f} MiB > --max-chunk-mib {max_chunk_mib}")
    with Dataset(out_path,"w",format="NETCDF4") as dst:
        dst.createDimension("time",T)
        dst.createDimension("point",len(point_ids))
        dst.createDimension("latitude",Y)
        dst.createDimension("longitude",X)
        for dim in dims:
            copy_coord(src,dst,dim)
        lat_idx=(point_ids//X).astype("i4")
        lon_idx=(point_ids%X).astype("i4")
        dst.createVariable("latitude_index","i4",("point",))[:]=lat_idx
        dst.createVariable("longitude_index","i4",("point",))[:]=lon_idx
        dst.setncattr("cudaheat_layout","time_point")
        dst.setncattr("original_variable",v.name)
        dst.setncattr("point_order","row_major_ocean_only")
        kwargs={"chunksizes":(T,min(chunk_points,len(point_ids))),
                "zlib":compression>0,"complevel":compression if compression>0 else 0,
                "shuffle":compression>0}
        if "_FillValue" in v.ncattrs():
            kwargs["fill_value"]=v.getncattr("_FillValue")
        target=dst.createVariable(v.name,v.dtype,("time","point"),**kwargs)
        for attr in v.ncattrs():
            if attr!="_FillValue":
                target.setncattr(attr,v.getncattr(attr))
        target.set_auto_maskandscale(False)  # Prevent double-packing raw SST integers.
        # Read complete time series for one row at a time. Raw packed ints retained.
        # Bound writer memory by output chunk size.
        for start in range(0,len(point_ids),chunk_points):
            ids=point_ids[start:start+chunk_points]
            ys=ids//X
            xs=ids%X
            data=np.empty((T,len(ids)),dtype=v.dtype)
            for y in np.unique(ys):
                positions=np.flatnonzero(ys==y)
                xvalues=xs[positions]
                # Contiguous row read avoids expensive repeated point reads.
                row=raw(v,(slice(None),int(y),slice(int(xvalues.min()),int(xvalues.max())+1)))
                data[:,positions]=row[:,xvalues-int(xvalues.min())]
            target[:,start:start+len(ids)]=data
    return chunk_mib

def read_original_batch(v, ids, T, X):
    ys=ids//X
    xs=ids%X
    out=np.empty((T,len(ids)),dtype=v.dtype)
    for y in np.unique(ys):
        positions=np.flatnonzero(ys==y)
        xx=xs[positions]
        row=raw(v,(slice(None),int(y),slice(int(xx.min()),int(xx.max())+1)))
        out[:,positions]=row[:,xx-int(xx.min())]
    return out

def timed_read_original(v, point_ids, shape, batch_points):
    T,_,X=shape
    tic=time.perf_counter()
    checksum=0
    for a in range(0,len(point_ids),batch_points):
        data=read_original_batch(v,point_ids[a:a+batch_points],T,X)
        checksum+=int(data.size)
    return time.perf_counter()-tic,checksum

def timed_read_optimized(v, npoints, batch_points, chunk_points):
    T=v.shape[0]
    effective=max(chunk_points,(batch_points//chunk_points)*chunk_points)
    tic=time.perf_counter()
    checksum=0
    for a in range(0,npoints,effective):
        data=raw(v,(slice(None),slice(a,min(npoints,a+effective))))
        checksum+=int(data.size)
    return time.perf_counter()-tic,checksum,effective

def verify(srcvar, optvar, point_ids, shape, n=12):
    T,_,X=shape
    rng=np.random.default_rng(123)
    picks=np.unique(np.r_[0,len(point_ids)-1,rng.integers(0,len(point_ids),size=min(n,len(point_ids)))])
    for p in picks:
        point=int(point_ids[p])
        a=raw(srcvar,(slice(None),point//X,point%X))
        b=raw(optvar,(slice(None),int(p)))
        if not np.array_equal(a,b,equal_nan=np.issubdtype(a.dtype,np.inexact)):
            raise AssertionError(f"Raw packed SST mismatch at point {p}")
    return len(picks)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--variable",default="analysed_sst")
    ap.add_argument("--chunks",default="4,8,16,24,32,48,64,96,128,192,256,512")
    ap.add_argument("--batch-points",default="64,128,256,512,1024,2048", help="Comma-separated simulated GPU batch sizes")
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--keep-optimized",action="store_true",help="Keep all optimized NetCDF variants; default deletes each after benchmarking")
    ap.add_argument("--scan-hours",type=int,default=24)
    ap.add_argument("--compression",type=int,default=5)
    ap.add_argument("--max-chunk-mib",type=float,default=256)
    ap.add_argument("--time-dim")
    ap.add_argument("--lat-dim")
    ap.add_argument("--lon-dim")
    args=ap.parse_args()
    if args.repeats<1 or args.scan_hours<1 or not 0<=args.compression<=9:
        ap.error("Invalid batch, repeats, scan-hours or compression")
    chunks=[int(x) for x in args.chunks.split(",")]
    batches=[int(x) for x in args.batch_points.split(",")]
    if any(b<1 for b in batches):
        ap.error("Batch sizes must be positive")
    if any(c<1 for c in chunks):
        ap.error("Chunk sizes must be positive")
    outdir=Path(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)
    report={"input":str(args.input),"settings":vars(args),"results":[]}
    with Dataset(args.input) as src:
        v,shape,dims=setup(src,args.variable,args)
        ids,land=scan_points(v,shape,args.scan_hours)
        report["dataset"]={"shape":shape,"original_points":shape[1]*shape[2],
                           "ocean_points":len(ids),"permanently_invalid_points":land,
                           "dtype":str(v.dtype),"original_chunking":v.chunking()}
        print("Dataset:",report["dataset"],flush=True)
        # Same ocean points in both cases: isolates layout/read-path changes.
        original_by_batch={}
        for batch in batches:
            times=[]
            for i in range(args.repeats):
                seconds,count=timed_read_original(v,ids,shape,batch)
                times.append(seconds)
                print(f"Original batch {batch}, repeat {i+1}: {seconds:.3f} s",flush=True)
            original_by_batch[str(batch)]=times
        report["original_ocean_only_seconds_by_batch"]=original_by_batch
        for c in chunks:
            target=outdir/f"sst_time_point_chunk_{c}.nc"
            print(f"Creating {target.name}",flush=True)
            try:
                size_mib=create_optimized(src,v,shape,dims,ids,target,c,args.compression,args.max_chunk_mib)
            except ValueError as exc:
                warnings.warn(str(exc))
                report["results"].append({"chunk_points":c,"skipped":str(exc)})
                continue
            with Dataset(target) as optimized:
                ov=optimized.variables[args.variable]
                checked=verify(v,ov,ids,shape)
                for batch in batches:
                    times=[]
                    for i in range(args.repeats):
                        seconds,count,effective=timed_read_optimized(ov,len(ids),batch,c)
                        times.append(seconds)
                        print(f"Chunk {c}, batch {batch}, repeat {i+1}: {seconds:.3f} s",flush=True)
                    original_median=float(np.median(original_by_batch[str(batch)]))
                    optimized_median=float(np.median(times))
                    result={"chunk_points":c,"optimized_file_retained":args.keep_optimized,"requested_batch_points":batch,
                            "chunk_uncompressed_mib":size_mib,
                            "file":str(target),"file_bytes":target.stat().st_size,
                            "verified_points":checked,"effective_batch_points":effective,
                            "read_seconds":times,"median_seconds":optimized_median,
                            "original_median_seconds":original_median,
                            "speedup_vs_original_ocean_only":original_median/optimized_median,
                            "raw_payload_mib":count*v.dtype.itemsize/2**20}
                    report["results"].append(result)
            (outdir/"benchmark_results.json").write_text(json.dumps(report,indent=2,default=str))
            if not args.keep_optimized:
                target.unlink()
                print(f"Removed temporary optimized file {target.name}",flush=True)
        (outdir/"benchmark_results.json").write_text(json.dumps(report,indent=2,default=str))
        print("Results:",outdir/"benchmark_results.json",flush=True)
        print("Note: OS/HDF5 caching affects repeats. These are warm-cache-prone, not cold-cache trials.")

if __name__=="__main__":
    main()
