import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, generic_filter
import sys

def create_25d_ortho(input_las_path, output_tif_path, resolution=0.03):
    print(f"Loading Point Cloud: {input_las_path}...")
    try:
        las = laspy.read(input_las_path)
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        sys.exit(1)

    # 1. Extract coordinates and colors
    x = las.x
    y = las.y
    z = las.z
    
    # Scale 16-bit colors down to 8-bit standard RGB (0-255)
    r = (las.red / 256).astype(np.uint8)
    g = (las.green / 256).astype(np.uint8)
    b = (las.blue / 256).astype(np.uint8)

    # 2. Define the Raster Grid Grid
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    
    print(f"Generating a {width} x {height} image at {resolution}m/pixel...")

    # Initialize empty layers
    z_grid = np.full((height, width), np.nan, dtype=np.float32)
    rgb_grid = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Map points to grid indices
    col_idx = ((x - xmin) / resolution).astype(int)
    row_idx = ((ymax - y) / resolution).astype(int)
    
    # Ensure indices are within bounds
    valid = (col_idx >= 0) & (col_idx < width) & (row_idx >= 0) & (row_idx < height)
    col_idx, row_idx = col_idx[valid], row_idx[valid]
    z_valid, r_valid, g_valid, b_valid = z[valid], r[valid], g[valid], b[valid]

    print("Populating RGB and Elevation layers...")
    # Advanced: Sort by Z descending so the highest points (curb edge) render on top of the gutter in the same pixel
    sort_idx = np.argsort(z_valid)
    row_idx, col_idx, z_valid = row_idx[sort_idx], col_idx[sort_idx], z_valid[sort_idx]
    r_valid, g_valid, b_valid = r_valid[sort_idx], g_valid[sort_idx], b_valid[sort_idx]

    # Populate the grids
    z_grid[row_idx, col_idx] = z_valid
    rgb_grid[row_idx, col_idx, 0] = r_valid
    rgb_grid[row_idx, col_idx, 1] = g_valid
    rgb_grid[row_idx, col_idx, 2] = b_valid

    # 3. Create the Shadow Layer (Slope/Gradient)
    print("Calculating physical shadows (Z-elevation drop-offs)...")
    # Fill small NoData holes in the Z-grid so the shadow calculation doesn't glitch
    mask = np.isnan(z_grid)
    z_grid[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), z_grid[~mask])
    
    # Calculate the gradient (slope) of the Z elevation
    dy, dx = np.gradient(z_grid, resolution, resolution)
    slope = np.sqrt(dx**2 + dy**2)
    
    # Normalize the slope to create a shadow mask (Steep drop = Dark, Flat = White)
    # We clip the slope so anything steeper than a 30% grade becomes solid black
    max_slope = 0.3 
    shadow_layer = 1.0 - np.clip(slope / max_slope, 0, 1)
    
    # 4. Multiply RGB by the Shadow Layer
    print("Baking 3D shadows into the 2D RGB image...")
    final_rgb = np.zeros_like(rgb_grid)
    for i in range(3):
        final_rgb[:, :, i] = (rgb_grid[:, :, i] * shadow_layer).astype(np.uint8)

    # 5. Export to GeoTIFF
    print("Exporting GeoTIFF...")
    transform = from_origin(xmin, ymax, resolution, resolution)
    
    with rasterio.open(
        output_tif_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3, # 3 bands for RGB
        dtype=final_rgb.dtype,
        crs="EPSG:3945", # Using your French projection from earlier!
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(final_rgb[:, :, 0], 1) # Red
        dst.write(final_rgb[:, :, 1], 2) # Green
        dst.write(final_rgb[:, :, 2], 3) # Blue

    print(f"✅ Masterpiece completed! Saved to {output_tif_path}")

# ==========================================
# Run the Script
# ==========================================
if __name__ == "__main__":
    # Change these paths to match your files!
    INPUT_CLOUD = "Laz/cloud.laz"
    OUTPUT_IMAGE = "training_ortho_3cm.tif"
    
    create_25d_ortho(INPUT_CLOUD, OUTPUT_IMAGE, resolution=0.03)