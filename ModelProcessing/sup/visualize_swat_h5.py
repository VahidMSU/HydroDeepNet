import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, BoundaryNorm
from multiprocessing import Process, Queue, Semaphore
from functools import partial
import time
import psutil
import logging

def generate_static_plots(f, group, NAME):
    figs_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_SWAT_gwflow_MODEL/watershed_static_plots"
    # If group is Landuse, read lookup table
    if group == "Landuse":
        lookup_table = f["Landuse/lookup_table"][()].decode("utf-8")

        # Parse lookup table into a dictionary
        lookup_dict = {}
        for line in lookup_table.strip().split("\n")[1:]:  # Skip header
            key, value = line.split(",")
            lookup_dict[int(key)] = value.strip()

        # Define a custom colormap for the landuse labels
        color_mapping = {
            "WATR": "#1f78b4",
            "WETF": "#a6cee3",
            "FRSD": "#33a02c",
            "FRSE": "#b2df8a",
            "FRST": "#7fbf7f",
            "AGRR": "#ff7f00",
            "HAY": "#fdbf6f",
            "URLD": "#6a3d9a",
            "URMD": "#cab2d6",
            "URHD": "#e31a1c",
            "UIDU": "#fb9a99",
            "SWRN": "#8b4513",
            "RNGE": "#d9d9d9",
            "RNGB": "#a6761d",
            "WETL": "#b3e2cd",
        }

        unique_labels = list(lookup_dict.values())
        colors = [color_mapping.get(label, "#999999") for label in unique_labels]  # Default to gray for missing labels
        cmap = ListedColormap(colors)
        bounds = list(lookup_dict.keys()) + [max(lookup_dict.keys()) + 1]
        norm = BoundaryNorm(bounds, cmap.N)
        tick_positions = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)]

    for key in f[group].keys():
        if os.path.exists(f"{figs_path}/{group}/{key}.png"):
            print(f"Static plot already exists for {key}")
            continue

        dataset = f[f"{group}/{key}"]
        if dataset.shape == ():  # Check if scalar
            print(f"Skipping scalar dataset: {group}/{key}")
            continue

        data = dataset[:]
        data = np.where(data < 0.0, np.nan, data)  # Replace negative values with NaN

        rows, cols = data.shape
        dpi = 100
        fig_size = (cols / dpi, rows / dpi)

        # Dynamically scale font size based on figure dimensions
        base_font_size = 10
        font_scale_factor = max(rows, cols) / 500
        scaled_font_size = base_font_size * font_scale_factor

        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        ax.grid(True, which='major', linestyle='--', linewidth=1.5)

        if group == "Landuse":
            mask = f["hru_wb_30m/2000/1/perc"][:]
            data = np.where(mask == -999, np.nan, data)

            img = ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
            colorbar = plt.colorbar(img, ax=ax, boundaries=bounds, ticks=tick_positions)
            colorbar.ax.tick_params(labelsize=scaled_font_size)
            colorbar.ax.set_yticklabels(unique_labels, fontsize=scaled_font_size)
            ax.set_title(f"{group} - {key}", fontsize=scaled_font_size * 1.2)
        else:
            img = ax.imshow(data, cmap="viridis", interpolation="nearest", aspect="auto")
            colorbar = plt.colorbar(img, ax=ax)
            colorbar.ax.tick_params(labelsize=scaled_font_size)
            ax.set_title(f"{group} - {key}", fontsize=scaled_font_size * 1.2)

        # Set x and y ticks and apply scaling
        xticks = np.arange(0, cols, max(1, cols // 5))
        yticks = np.arange(0, rows, max(1, rows // 5))
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', which='major', labelsize=scaled_font_size*0.8)
        

        os.makedirs(f"{figs_path}/{group}", exist_ok=True)
        fig_path = f"{figs_path}/{group}/{key}.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved static plot for {key}: {fig_path}")
def generate_spatiotemporal_animation(f, var_name, NAME, ver):
    years = range(2000, 2020)
    months = range(1, 13)
    mask = f[f"hru_wb_30m/2000/1/perc"][:]
    DEM = f[f"DEM/dem"][:]
    assert mask.shape == DEM.shape, f"Mask and DEM shapes do not match: {mask.shape} != {DEM.shape}"
    video_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_SWAT_gwflow_MODEL/verifications_videos"

    rows, cols = mask.shape
    dpi = 100
    fig_size = (cols / dpi, rows / dpi)
    base_font_size = 10
    font_scale_factor = max(rows, cols) / 500
    scaled_font_size = base_font_size * font_scale_factor

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.grid(True, which='major', linestyle='--', linewidth=1)
    img = ax.imshow(np.zeros((rows, cols)), cmap="viridis", interpolation="nearest", aspect='auto')
    colorbar = plt.colorbar(img, ax=ax)
    colorbar.ax.tick_params(labelsize=scaled_font_size)
    title = ax.set_title("", fontsize=scaled_font_size * 1.2)

    xticks = np.arange(0, cols, max(1, cols // 5))
    yticks = np.arange(0, rows, max(1, rows // 5))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(axis='both', which='major', labelsize=scaled_font_size*0.8)
    

    def update(frame):
        year, month = frame
        data = f[f"hru_wb_30m/{year}/{month}/{var_name}"][:]
        data = np.where(data < 0.0, np.nan, data)
        _97_5pth = np.nanpercentile(data, 97.5)
        _2_5pth = np.nanpercentile(data, 2.5)
        data = np.clip(data, _2_5pth, _97_5pth)
        img.set_data(data)
        img.set_clim(vmin=np.nanmin(data), vmax=np.nanmax(data))
        month_name = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        title.set_text(f"{var_name.capitalize()} - Year: {year}, Month: {month_name[month]}")
        return img, title

    frames = [(year, month) for year in years for month in months]
    ani = FuncAnimation(fig, update, frames=frames, interval=200, blit=False)

    os.makedirs(f"{video_path}", exist_ok=True)
    gif_path = f"{video_path}/{ver}_{var_name}_animation.gif"
    ani.save(gif_path, writer=PillowWriter(fps=5, metadata={'title': var_name}), dpi=dpi)
    print(f"Animation saved: {gif_path}")
    plt.close(fig)


def ram_usage():
		# Get memory details
		memory = psutil.virtual_memory()
		# Total memory
		total_memory = memory.total / (1024 ** 3)  # Convert bytes to GB
		logging.info(f"Total Memory: {total_memory:.2f} GB")
		# Used memory
		used_memory = memory.used / (1024 ** 3)  # Convert bytes to GB
		logging.info(f"Used Memory: {used_memory:.2f} GB")
		# Memory usage percentage
		memory_usage_percent = memory.percent
		logging.info(f"Memory Usage: {memory_usage_percent}%")

		return used_memory

def visualize_swatplus_h5(NAME="04136000", ver=0):
    path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/verification_stage_{ver}/SWATplus_output.h5"
    figs_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_SWAT_gwflow_MODEL/watershed_static_plots"
    video_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_SWAT_gwflow_MODEL/verifications_videos"
    # Create directories for outputs
    os.makedirs(f"{video_path}", exist_ok=True)
    os.makedirs(f"{figs_path}", exist_ok=True)
    while not ram_usage() < 300:
        print("CPU usage is high, waiting for 2 minutes")
        time.sleep(120)
    # Open HDF5 file
    with h5py.File(path, "r") as f:
        # Time-varying variable animation
        processes = []  
        ## if already exists, skip
        if os.path.exists(f"{video_path}/{ver}_et_animation.gif"):
            print(f"Animation already exists for {NAME}, verification stage {ver}")
        else:
            for var_name in ["et", "perc", "precip", "snofall", "snomlt", "surq_gen", "wateryld"]:
                p = Process(target=generate_spatiotemporal_animation, args=(f, var_name, NAME, ver))
                p.start()
                time.sleep(10)
                while not ram_usage() < 300:
                    print("CPU usage is high, waiting for 2 minutes")
                    time.sleep(120)
                processes.append(p)
            for p in processes:
                p.join()
        # Static soil data plots
        group = "Soil"
        generate_static_plots(f, group, NAME)
        # Static DEM data plots
        group = "DEM"
        generate_static_plots(f, group, NAME)
        # Static Landuse data plots
        group = "Landuse"
        generate_static_plots(f, group, NAME)


def worker_process(sem, name, ver):
    """Worker function to visualize SWATplus output."""
    try:
        visualize_swatplus_h5(NAME=name, ver=ver)
    finally:
        # Release the semaphore after the process completes
        sem.release()
if __name__ == "__main__":
    NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12")
    NAMES.remove("log.txt")
    n_workers = 12  # Number of worker processes
    sem = Semaphore(n_workers)  # Semaphore to control active processes
    processes = []

    # Create a queue with tasks
    tasks = Queue()
    for NAME in NAMES:
        for ver in range(6):
            tasks.put((NAME, ver))

    # Start worker processes
    while not tasks.empty():
        sem.acquire()  # Acquire a semaphore slot before starting a process
        NAME, ver = tasks.get()  # Get the next task
        p = Process(target=worker_process, args=(sem, NAME, ver))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("All tasks completed.")