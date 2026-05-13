from pathlib import Path
import numpy as np
import rasterio
import sys

from tqdm import tqdm
from geosat.yoloroad.lrm_generator import msrm


def save_lrm_geotiff(
    output_path: Path, lrm_array: np.ndarray, reference_tiff: Path, nodata: float
):
    """
    Saves a Local Relief Model (LRM) array as a GeoTIFF using a reference TIFF for metadata.
    """
    with rasterio.open(reference_tiff) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=nodata, compress="lzw")

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(lrm_array.astype(np.float32), 1)


def compute_and_save_lrm(
    input_tiff: Path,
    resolution: float,
    output_folder: Path,
    feature_min: float,
    feature_max: float,
    scaling_factor: float,
    ve_factor: float,
    tile_size: int = 8192,  # 2^13 to balance memory efficiency and computational speed.
) -> Path:
    """
    Computes the Local Relief Model from a DTM by processing it in tiles to avoid memory issues.
    """
    try:
        input_path = input_tiff.resolve()
        output_folder = output_folder.resolve()
        output_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error preparing paths: {e}", file=sys.stderr)
        sys.exit(1)

    output_lrm_tif = output_folder / f"{input_path.stem}_lrm.tif"

    # Load the DTM with rasterio to allow block-wise reading
    with rasterio.open(input_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        width, height = src.width, src.height
        lrm_full = np.full((height, width), nodata, dtype=np.float32)

        y_steps = list(range(0, height, tile_size))
        x_steps = list(range(0, width, tile_size))

        print("Processing DTM in tiles...")
        for y in tqdm(y_steps, desc="Rows"):
            for x in x_steps:
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                window = rasterio.windows.Window(x, y, x_end - x, y_end - y)
                dtm_tile = src.read(1, window=window).astype(np.float32)

                # Compute LRM for the tile
                try:
                    lrm_tile = -msrm(
                        dem=dtm_tile,
                        resolution=resolution,
                        feature_min=feature_min,
                        feature_max=feature_max,
                        scaling_factor=scaling_factor,
                        ve_factor=ve_factor,
                        no_data=nodata,
                    )
                except Exception as e:
                    print(
                        f"Error computing LRM tile at ({x},{y}): {e}", file=sys.stderr
                    )
                    lrm_tile = np.full(dtm_tile.shape, nodata, dtype=np.float32)

                # Copy the tile into the final array
                lrm_full[y:y_end, x:x_end] = lrm_tile

    # Save as GeoTIFF
    save_lrm_geotiff(output_lrm_tif, lrm_full, input_path, nodata)
    print(f"✅ LRM saved: {output_lrm_tif}")
    return output_lrm_tif


if __name__ == "__main__":

    # ------------------ Define your input/output folders ------------------
    input_folder = Path(
        "DTM/"
    )  # Example input folder containing DTM GeoTIFFs
    output_folder = Path(
        "LRM/"
    )  # Example output folder for generated LRMs

    # ------------------ Default LRM parameters ------------------
    resolution = 0.1  # Spatial resolution in meters/pixel
    feature_min = 0.1  # Minimum feature size for relief computation
    feature_max = 2.5  # Maximum feature size for relief computation
    scaling_factor = 1.5  # Scaling factor applied during LRM computation
    ve_factor = 2  # Vertical exaggeration factor

    # ------------------ Check input folder existence ------------------
    if not input_folder.exists():
        print(f"❌ Input folder not found: {input_folder}")
        sys.exit(1)

    # ------------------ Ensure output folder exists ------------------
    output_folder.mkdir(parents=True, exist_ok=True)

    # ------------------ List all DTM GeoTIFF files ------------------
    dtm_files = sorted(input_folder.glob("*.tif"))
    if not dtm_files:
        print(f"⚠️ No DTM files found in {input_folder}")
        sys.exit(0)

    # ------------------ Process each DTM ------------------
    for dtm in dtm_files:
        print(f"➡️ Processing {dtm.name}...")
        compute_and_save_lrm(
            dtm,
            resolution,
            output_folder,
            feature_min,
            feature_max,
            scaling_factor,
            ve_factor,
        )
