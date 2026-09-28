import rasterio
from rasterio.enums import Resampling

gdb_path = "path_to_your_database.gdb"
raster_layer_name = "your_raster_layer_name"
output_tiff = "downscaled_500m.tif"

gdb_raster_path = f"gdb://{gdb_path}:{raster_layer_name}"

downscale_factor = 10 / 500

with rasterio.open(gdb_raster_path) as src:
       grid = data[0]
    nodata_value = src.nodata

    if nodata_value is not None:
        valid_mask = (grid != nodata_value) & (~np.isnan(grid))
        null_count = np.sum((grid == nodata_value) | np.isnan(grid))
    else:
        valid_mask = ~np.isnan(grid)
        null_count = np.sum(np.isnan(grid))

    # Calculate and print stats if valid data exists
    if np.any(valid_mask):
        valid_data = grid[valid_mask]
        print("----- DIAGNOSTICS -----")
        print(f"Total Pixels: {grid.size:,}")
        print(f"Null/NoData Pixels: {null_count:,} ({null_count / grid.size * 100:.2f}%)")
        print(f"Minimum Depth: {np.min(valid_data)}")
        print(f"Maximum Depth: {np.max(valid_data)}")
        print("-----------------------")
    else:
        print(f"Warning: Grid contains 100% Null/NoData values ({grid.size:,} pixels).")
        
    new_height = int(src.height * downscale_factor)
    new_width = int(src.width * downscale_factor)
    
    data = src.read(
        out_shape=(src.count, new_height, new_width),
        resampling=Resampling.average 
    )
    
    transform = src.transform * src.transform.scale(
        (src.width / data.shape[-1]),
        (src.height / data.shape[-2])
    )
    
    meta = src.meta.copy()
    meta.update({
        'driver': 'GTiff',
        'height': new_height,
        'width': new_width,
        'transform': transform
    })
    
    with rasterio.open(output_tiff, 'w', **meta) as dst:
        dst.write(data)

print(f"Successfully downscaled and saved to {output_tiff}")
