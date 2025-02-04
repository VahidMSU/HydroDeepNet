import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_static_video(f, group, NAME):
    """
    Generate static plots for soil, DEM, and landuse as PNGs suitable for embedding in HTML.
    """
    for key in f[group].keys():
        print(f"Processing {group} - {key} - {NAME}")
        data = f[f"{group}/{key}"][:]
        data = np.where(data < 0.0, np.nan, data)

        # Generate the static plot
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap="viridis", interpolation="nearest")
        plt.colorbar(label=f"{group.capitalize()} - {key}")
        plt.title(f"{group.capitalize()} - {key}")
        os.makedirs(f"video/{NAME}/{group}", exist_ok=True)
        png_path = f"video/{NAME}/{group}/{key}.png"
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()
        print(f"Static plot saved: {png_path}")

def generate_spatiotemporal_video(f, var_name, NAME, ver):
    """
    Generate videos for time-varying variables suitable for embedding in HTML.
    """
    years = range(2000, 2020)
    months = range(1, 13)

    mask = f[f"hru_wb_30m/2000/1/perc"][:]
    rows, cols = mask.shape
    dpi = 100
    fig_size = (cols / dpi, rows / dpi)

    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    img = ax.imshow(np.zeros((rows, cols)), cmap="viridis", interpolation="nearest", aspect='auto')
    colorbar = plt.colorbar(img, ax=ax)
    title = ax.set_title("")

    def update(frame):
        year, month = frame
        print(f"Processing year {year}, month {month}, variable {var_name}, NAME {NAME}, verification stage {ver}")
        data = f[f"hru_wb_30m/{year}/{month}/{var_name}"][:]
        data = np.where(data < 0.0, np.nan, data)
        _97_5pth = np.nanpercentile(data, 97.5)
        _2_5pth = np.nanpercentile(data, 2.5)
        # Clamp data to 97.5th and 2.5th percentiles
        data = np.where(data > _97_5pth, _97_5pth, data)
        data = np.where(data < _2_5pth, _2_5pth, data)
        img.set_data(data)
        img.set_clim(vmin=_2_5pth, vmax=_97_5pth)  # Set dynamic color limits
        title.set_text(f"{var_name.capitalize()} - Year: {year}, Month: {month}")
        return img, title

    # Generate frames for all year-month combinations
    frames = [(year, month) for year in years for month in months]
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=False)

    # Save the video
    os.makedirs(f"video/{NAME}", exist_ok=True)
    video_path = f"video/{NAME}/{ver}_{var_name}_animation.mp4"
    ani.save(video_path, writer="ffmpeg", fps=5)
    print(f"Video saved: {video_path}")
    plt.close(fig)

def process_swatplus_output(NAME, ver, var_name):
    """
    Process the SWAT+ output to generate videos and static plots.
    """
    path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/verification_stage_{ver}/SWATplus_output.h5"
    with h5py.File(path, "r") as f:
        # Generate video for spatiotemporal data
        generate_spatiotemporal_video(f, var_name, NAME, ver)
        # Generate static PNGs for Soil, DEM, and Landuse
        generate_static_video(f, "Soil", NAME)
        generate_static_video(f, "DEM", NAME)
        generate_static_video(f, "Landuse", NAME)

if __name__ == "__main__":
    """
    Main function to process multiple variables, names, and verification stages.
    """
    NAMES = os.listdir("/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12")
    NAMES.remove("log.txt")
    for var_name in ["et", "perc", "precip", "snofall", "snomlt", "surq_gen", "wateryld"]:
        for NAME in NAMES:
            for ver in range(0, 6):
                process_swatplus_output(NAME, ver, var_name)
                break
            break
        break
