"""Run the first CUDAHeat preprocessing milestone on a NetCDF SST variable."""

from cudaheat import CUDAHeat


def main() -> None:
    with CUDAHeat(
        "sst.nc", variable="analysed_sst", dtype="float32", vram_safety=0.30, devices=[0]
    ) as hw:
        result = hw.run(baseline=(1982, 2011), percentile=0.90, smooth_window_hours=120)
        print(f"Processed {hw.ntime} hours and {hw.npoints} points")
        print(result.exceedance_count)
        for batch in result.telemetry:
            print(batch.as_dict())


if __name__ == "__main__":
    main()
