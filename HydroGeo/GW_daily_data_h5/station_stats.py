import h5py
import numpy as np
import os

def read_h5(h5_path):
    stds = []
    means = []
    medians = []
    ranges = []
    iqrs = []
    cvs = []

    if os.path.exists("stats.txt"):
        os.remove("stats.txt")

    with h5py.File(h5_path, "r") as h5_file:
        stations = list(h5_file.keys())
        for station in stations:
            data = h5_file[station][:]
            lat = h5_file[station].attrs["lat"]
            lon = h5_file[station].attrs["lon"]
            # Drop NaN values
            data = data[~np.isnan(data)]

            # Standard deviation
            std = np.std(data)
            stds.append(std)

            # Mean
            mean = np.mean(data)
            means.append(mean)

            # Median
            median = np.median(data)
            medians.append(median)

            # Range
            data_range = np.max(data) - np.min(data)
            ranges.append(data_range)

            # Interquartile Range (IQR)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            iqrs.append(iqr)

            # Coefficient of Variation (CV)
            cv = std / mean if mean != 0 else np.nan
            cvs.append(cv)

            with open("stats.txt", "a") as f:
                f.write(f"{station} lat: {lat}, lon: {lon}, std: {std:.2f}, mean: {mean:.2f}, median: {median:.2f}, range: {data_range:.2f}, IQR: {iqr:.2f}, CV: {cv:.2f}\n")

    # Calculate and print averages
    avg_std = np.mean(stds)
    avg_mean = np.mean(means)
    avg_median = np.mean(medians)
    avg_range = np.mean(ranges)
    avg_iqr = np.mean(iqrs)
    avg_cv = np.nanmean(cvs)  # Use nanmean to ignore NaNs

    max_std = np.max(stds)
    max_mean = np.max(means)
    max_median = np.max(medians)
    max_range = np.max(ranges)
    max_iqr = np.max(iqrs)
    max_cv = np.nanmax(cvs)

    median_std = np.median(stds)
    median_mean = np.median(means)
    median_median = np.median(medians)
    median_range = np.median(ranges)
    median_iqr = np.median(iqrs)
    median_cv = np.nanmedian(cvs)

    print(f"Average standard deviation: {avg_std:.2f}")
    print(f"Average mean: {avg_mean:.2f}")
    print(f"Average median: {avg_median:.2f}")
    print(f"Average range: {avg_range:.2f}")
    print(f"Average IQR: {avg_iqr:.2f}")
    print(f"Average CV: {avg_cv:.2f}")

    if os.path.exists("stats.txt"):
        with open("stats.txt", "a") as f:
            f.write("\nOverall Statistics\n")
            f.write("------------------\n")
            f.write(f"{'Statistic':<25}{'Value':>10}\n")
            f.write(f"{'Average standard deviation:':<25}{avg_std:.2f}\n")
            f.write(f"{'Average mean:':<25}{avg_mean:.2f}\n")
            f.write(f"{'Average median:':<25}{avg_median:.2f}\n")
            f.write(f"{'Average range:':<25}{avg_range:.2f}\n")
            f.write(f"{'Average IQR:':<25}{avg_iqr:.2f}\n")
            f.write(f"{'Average CV:':<25}{avg_cv:.2f}\n")
            f.write("\n")
            f.write(f"{'Max standard deviation:':<25}{max_std:.2f}\n")
            f.write(f"{'Max mean:':<25}{max_mean:.2f}\n")
            f.write(f"{'Max median:':<25}{max_median:.2f}\n")
            f.write(f"{'Max range:':<25}{max_range:.2f}\n")
            f.write(f"{'Max IQR:':<25}{max_iqr:.2f}\n")
            f.write(f"{'Max CV:':<25}{max_cv:.2f}\n")
            f.write("\n")
            f.write(f"{'Median standard deviation:':<25}{median_std:.2f}\n")
            f.write(f"{'Median mean:':<25}{median_mean:.2f}\n")
            f.write(f"{'Median median:':<25}{median_median:.2f}\n")
            f.write(f"{'Median range:':<25}{median_range:.2f}\n")
            f.write(f"{'Median IQR:':<25}{median_iqr:.2f}\n")
            f.write(f"{'Median CV:':<25}{median_cv:.2f}\n")

if __name__ == "__main__":
    read_h5("gw_head_2d.h5")
