"""Run the first CUDAHeat preprocessing milestone on a NetCDF SST variable."""

from cudaheat import CUDAHeat

PATH = "/home/user/Documents/PhD.cy/lazy_loading/Python.Scripts/sst_7_years.nc"
def main() -> None:
    with CUDAHeat(
        PATH, variable="analysed_sst", dtype="float32", vram_safety=0.30, devices=[0]
    ) as hw:
        result = hw.run(baseline=(2019, 2023), percentile=0.90, smooth_window_hours=120)
        print(f"Processed {hw.ntime} hours and {hw.npoints} points")
        print(result.exceedance_count)
        for batch in result.telemetry:
            print(batch.as_dict())
            print()

        count = result.exceedance_count

        print("\nExceedance statistics")
        print("---------------------")
        print("Minimum:", count.min().item())
        print("Maximum:", count.max().item())
        print("Mean:   ", count.mean().item())
        print("Nonzero:", (count > 0).sum().item())
        print("Total:  ", count.size)

if __name__ == "__main__":
    main()
