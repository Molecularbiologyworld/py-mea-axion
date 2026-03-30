from mea_raster_generator import run

run(
    input_file  = r"C:\path\to\your_recording.spk",
    output_dir  = r"C:\path\to\output_folder",
    wells       = ["D1"],      # e.g. ["A1", "B2"] or None / ["ALL"] for every well
    time_start  = 0,           # seconds
    time_end    = 420,         # seconds (0 = full recording)
    asdr_thresh = 50,          # red dashed threshold line
    combined    = True,        # True = raster + ASDR; False = ASDR only
    asdr_y_max  = None,        # None = autoscale; e.g. 120 to fix the axis
    dpi         = 300,
    # --- only needed for .raw or .npz files ---
    rec_seconds = 0,
    bin_ms      = 200,
    thresh_k    = 5.0,
)
