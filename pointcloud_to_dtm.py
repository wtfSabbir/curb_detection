from pathlib import Path
from typing import Tuple
import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm
import sys


def read_las_points(
    las_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, laspy.LasHeader]:
    """
    Read the XYZ coordinates and header from a LAS/LAZ point cloud file.

    Args:
        las_path (Path): Path to the LAS/LAZ file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, laspy.LasHeader]: x, y, z coordinates and LAS header.
    """
    las = laspy.read(las_path)
    header = las.header
    x = las.x
    y = las.y
    z = las.z
    return x, y, z, header


def create_dtm_raster(
    output_file: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    resolution: float = 0.05,
) -> Tuple[np.ndarray, rasterio.transform.Affine]:
    """
    Create a Digital Terrain Model (DTM) raster from XYZ point cloud data.

    Args:
        output_file (Path): Path to the output raster (used for progress display).
        x, y, z (np.ndarray): Coordinates of the point cloud.
        resolution (float): Raster resolution in the same units as coordinates.

    Returns:
        Tuple[np.ndarray, Affine]: DTM raster array and its affine transform.
    """
    try:
        # Compute raster bounds
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        ncols = int(np.ceil((xmax - xmin) / resolution))
        nrows = int(np.ceil((ymax - ymin) / resolution))

        # Compute pixel indices
        ix = ((x - xmin) / resolution).astype(int)
        iy = ((ymax - y) / resolution).astype(int)

        # Initialize raster with NaN
        raster = np.full((nrows, ncols), np.nan, dtype=np.float32)

        # Populate raster with minimum Z value per cell
        for i in tqdm(range(len(z)), desc=f"Building DTM: {output_file.name}"):
            col, row = ix[i], iy[i]
            if 0 <= row < nrows and 0 <= col < ncols:
                current_val = raster[row, col]
                raster[row, col] = (
                    z[i] if np.isnan(current_val) else min(current_val, z[i])
                )

        transform = from_origin(xmin, ymax, resolution, resolution)
        return raster, transform

    except Exception as e:
        print(f"Error creating DTM raster: {e}", file=sys.stderr)
        sys.exit(1)


def save_raster(
    raster: np.ndarray,
    transform: rasterio.transform.Affine,
    output_file: Path,
    crs,
    nodata: float = np.nan,
) -> None:
    """
    Save a raster array to a GeoTIFF file with a specified CRS and affine transform.

    Args:
        raster (np.ndarray): 2D raster array.
        transform (Affine): Affine transform for georeferencing.
        output_file (Path): Path to the output GeoTIFF.
        crs: Coordinate reference system (EPSG code or CRS object).
        nodata (float): NoData value.
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory for raster: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=raster.dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
            nodata=nodata,
        ) as dst:
            dst.write(raster, 1)
    except Exception as e:
        print(f"Error saving raster to file {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


def laz_to_dtm(
    pointcloud_path: Path, resolution: float, output_folder: Path, crs: int
) -> Path:
    """
    Convert a LAS/LAZ point cloud into a DTM raster and optionally generate a grayscale ortho image.

    Args:
        pointcloud_path (Path): Input point cloud file.
        resolution (float): Raster resolution.
        output_folder (Path): Folder to save outputs.
        crs (int): EPSG code of coordinate system.

    Returns:
        Path: Path to the saved DTM raster.
    """
    pointcloud_path = pointcloud_path.resolve()
    output_folder = output_folder.resolve()

    try:
        output_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output folder: {e}", file=sys.stderr)
        sys.exit(1)

    point_count = laspy.read(pointcloud_path).header.point_count
    if point_count == 0:
        print(f"⚠️  Empty filtered point cloud ({pointcloud_path.name}), file skipped.")
        return Path("")

    x, y, z, header = read_las_points(pointcloud_path)

    # Determine unique output filename
    dtm_stem = pointcloud_path.stem + "_dtm"
    dtm_path = output_folder / f"{dtm_stem}.tif"
    counter = 1
    while dtm_path.exists():
        dtm_path = output_folder / f"{dtm_stem}_{counter}.tif"
        counter += 1

    raster, transform = create_dtm_raster(dtm_path, x, y, z, resolution)
    save_raster(raster, transform, dtm_path, crs)

    print(f"✅ DTM saved to: {dtm_path}")

    # Optionally generate an ortho grayscale image from RGB or intensity attributes
    las = laspy.read(pointcloud_path)
    attributes = {}
    for attr in ["red", "green", "blue", "intensity"]:
        if hasattr(las, attr):
            attributes[attr] = getattr(las, attr)

    if attributes:
        ortho_stem = dtm_path.stem.replace("_dtm", "_ortho")
        ortho_path = output_folder / f"{ortho_stem}.tif"
        counter = 1
        while ortho_path.exists():
            ortho_path = output_folder / f"{ortho_stem}_{counter}.tif"
            counter += 1
        # create_ortho_image(ortho_path, x, y, attributes, resolution, crs)
    else:
        print("⚠️ No RGB or intensity attributes available for ortho image.")

    return dtm_path


def create_ortho_image(
    output_file: Path,
    x: np.ndarray,
    y: np.ndarray,
    attributes: dict,
    resolution: float,
    crs,
) -> None:
    """
    Generate a normalized grayscale ortho image from point cloud attributes (RGB or intensity).

    Args:
        output_file (Path): Path to save the ortho image.
        x, y (np.ndarray): Coordinates of the point cloud.
        attributes (dict): Point cloud attributes (red, green, blue, intensity).
        resolution (float): Pixel resolution.
        crs: EPSG code or CRS object for output image.
    """
    print("Generating grayscale raster...")

    # Define raster bounds
    xmin = np.floor(x.min() / resolution) * resolution
    xmax = np.ceil(x.max() / resolution) * resolution
    ymin = np.floor(y.min() / resolution) * resolution
    ymax = np.ceil(y.max() / resolution) * resolution

    ncols = int(round((xmax - xmin) / resolution))
    nrows = int(round((ymax - ymin) / resolution))

    # Compute pixel indices
    ix = ((x - xmin) / resolution).astype(int)
    iy = ((ymax - y) / resolution).astype(int)

    gray = np.zeros((nrows, ncols), dtype=np.uint8)

    # Use RGB attributes if available
    if {"red", "green", "blue"}.issubset(attributes.keys()):
        rgb_scale = (
            256
            if (
                attributes["red"].max() > 255
                or attributes["green"].max() > 255
                or attributes["blue"].max() > 255
            )
            else 1
        )

        for i in range(len(x)):
            row, col = iy[i], ix[i]
            if 0 <= row < nrows and 0 <= col < ncols:
                r = attributes["red"][i] // rgb_scale
                g = attributes["green"][i] // rgb_scale
                b = attributes["blue"][i] // rgb_scale
                gray[row, col] = int(0.2989 * r + 0.5870 * g + 0.1140 * b)

    # Use intensity if RGB not available
    elif "intensity" in attributes:
        temp_intensity = np.zeros((nrows, ncols), dtype=np.float32)
        for i in range(len(x)):
            row, col = iy[i], ix[i]
            if 0 <= row < nrows and 0 <= col < ncols:
                temp_intensity[row, col] = attributes["intensity"][i]

        # Normalize intensity to 0-255
        min_val, max_val = temp_intensity.min(), temp_intensity.max()
        if max_val > min_val:
            gray = ((temp_intensity - min_val) / (max_val - min_val) * 255).astype(
                np.uint8
            )

    else:
        print("⚠️ No RGB or intensity information available to generate ortho image.")
        return

    # Apply global histogram stretching
    min_gray, max_gray = gray.min(), gray.max()
    if max_gray > min_gray:
        gray = ((gray - min_gray) / (max_gray - min_gray) * 255).astype(np.uint8)

    transform = from_origin(xmin, ymax, resolution, resolution)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save raster
    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(gray, 1)

    print(f"✅ Normalized grayscale ortho image saved: {output_file}")


def process_pointcloud_folder(
    input_folder: Path, output_folder: Path, resolution: float, crs: int
):
    """
    Parcourt tous les fichiers LAS/LAZ d'un dossier et génère des DTM dans un autre dossier.

    Args:
        input_folder (Path): Dossier contenant les fichiers .las / .laz
        output_folder (Path): Dossier où sauvegarder les DTM
        resolution (float): Résolution du raster
        crs (int): EPSG code du système de coordonnées
    """
    input_folder = input_folder.resolve()
    output_folder = output_folder.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(input_folder.glob("*.la[sz]"))
    if not files:
        print(f"⚠️ Aucun fichier .las/.laz trouvé dans {input_folder}")
        return

    for f in files:
        print(f"➡️ Traitement de {f.name}...")
        try:
            laz_to_dtm(f, resolution, output_folder, crs)
        except Exception as e:
            print(f"❌ Erreur avec {f.name}: {e}")


if __name__ == "__main__":

    # Example usage: process all LAS/LAZ files in an input folder to generate DTMs and optional ortho images.

    # Define your input and output folders here
    input_folder = Path("Laz/")
    output_folder = Path("DTM/")

    # Default parameters
    resolution = (
        0.05  # Raster resolution in the same units as point cloud (e.g., meters)
    )
    crs_epsg = 3947  # EPSG code for output rasters

    # Check if input folder exists
    if not input_folder.exists():
        print(f"❌ Input folder not found: {input_folder}")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    # Process all LAS/LAZ files
    process_pointcloud_folder(input_folder, output_folder, resolution, crs_epsg)

    print("✅ All point clouds have been processed.")
