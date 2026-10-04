from cudaheat import CUDAHeat

PATH = "/home/user/Documents/PhD.cy/lazy_loading/Python.Scripts/sst_7_years.nc"

with CUDAHeat(
    PATH,
    variable="analysed_sst",
    dtype="float32",
    vram_safety=0.710,
    devices=[0],
) as hw:

    print("\nCUDAHeat dataset")
    print("----------------")
    print("Time steps:     ", hw.ntime)
    print("Spatial points: ", hw.npoints)
    print("Batch size:     ", hw.batch_size)
    print("GPU info:       ", hw.gpu_info)

    result = hw.run()