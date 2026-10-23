import xarray as xr
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt


PATH = "/home/user/Documents/Gea/cudaheat/benchmarks/extend_NETCDF_create40_years_timespan/sst_40_years.nc"

ds = xr.open_dataset(PATH, chunks="auto")["analysed_sst"]





# lon, lat = 32, 34

# pts = ds.sel(longitude=lon, latitude= lat, method="nearest")
# print(pts)

# pts.plot()
# plt.show()