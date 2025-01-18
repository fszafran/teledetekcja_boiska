import subprocess
import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd

def gen_ndti(raster_path, output_path):
    with rasterio.open(raster_path) as src:
        green = src.read(3)
        nir = src.read(8)
        ndti = np.where((green + nir) == 0, 0, (green - nir) / (green + nir))
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,
            count=1)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(ndti, 1)
    return ndti

def calc_min_max_with_polygons(ndti_path, polygons_path, val_tolerance=1):
    polygons = gpd.read_file(polygons_path)
    with rasterio.open(ndti_path) as src:
        polygons = polygons.to_crs(src.crs)

    objects = []
    for i in range(len(polygons)):
        object_geometry = [polygons.iloc[i].geometry]
        with rasterio.open(ndti_path) as src:
            out_image, _ = mask(src, object_geometry, crop=True, nodata=np.nan)
            objects.append(out_image)

    values = []
    for obj in objects:
        min_val = np.nanpercentile(obj, 15)
        max_val = np.nanmax(obj)
        values.append((min_val, max_val))
        print(min_val, max_val)

    return np.min([x[0] for x in values]) - val_tolerance, np.max([x[1] for x in values]) + val_tolerance


def extract_mask_by_range(ndti_path, min_val, max_val, output_path):
    with rasterio.open(ndti_path) as src:
        data = src.read(1)

        # Create a mask for values within the range
        mask = (data >= min_val) & (data <= max_val)

        # Optionally create a new array with only values in range, others set to nodata
        filtered_data = np.where(mask, data, src.nodata)

        profile = src.profile
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(filtered_data, 1)
    return data

if __name__ == '__main__':
    img_path = 'copy_im.tif'
    ndti_path = 'rasters/ndti.tif'
    polygons_path = 'poligony/polyg.shp'
    filtered_path = 'rasters/filtered_raster2.tif'
    candidates_path = 'poligony/candidates.shp'

    # gen_ndti(img_path, ndti_path)
    # min_value, max_value = calc_min_max_with_polygons(ndti_path, polygons_path)
    # print()
    # print(min_value, max_value)
    min_value, max_value = 26, 30
    extract_mask_by_range(ndti_path, min_value, max_value, filtered_path)


    '''
    TODO:
    31 50.5

    Raster mask
    Reclasify !Nan -> 1
    Raster to polygon
    Selekcja area 1600-2000
    BBOX
    selekcja proporcje i area
    
    Shape_Area / Shape_Length  > 9.95 AND Shape_Area / Shape_Length < 10.6 AND Shape_Area > 1600 AND Shape_Area < 2000
    
    '''