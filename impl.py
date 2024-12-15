import numpy as np
from collections import deque
from typing import Set, Tuple, List, Union
from pathlib import Path
import cv2
from shapely.geometry import Polygon
import osgeo.gdal as gdal
gdal.UseExceptions()

class Params:
    def __init__(self, image: np.ndarray, pixel_size: int, min_val: int, top_val: int, model_length: int, model_width: int):
        self.image = image
        self.pixel_size = pixel_size
        self.min_val = min_val
        self.top_val = top_val
        self.model_length = model_length
        self.model_width = model_width
        self.model_area = model_length * model_width
        self.model_perimeter = 2*(model_length+model_width)

def read_spatial_raster(path: Union[str, Path]) -> gdal.Dataset:
    dataset = gdal.Open(str(path))
    assert dataset is not None, "Read spatial raster returned None"
    return dataset

def read_raster_bands(dataset: gdal.Dataset, band_indices: List[int]) -> List[gdal.Band]:
    bands = []
    for index in band_indices:
        band = dataset.GetRasterBand(index)
        assert band is not None, f"Band {index} is not available"
        bands.append(band)
    return bands

def read_band_as_array(band: gdal.Band) -> np.ndarray:
    array = band.ReadAsArray()
    array = np.copy(array) 
    return array

def pixel_value_in_range(pixel_value: int , min_value: int, top_value: int) -> bool:
    return min_value <= pixel_value <= top_value

def neighbours(image: np.ndarray, s_row: int , s_col: int) -> List[Tuple[int, int]]:
    indices = [(s_row-1, s_col), (s_row+1, s_col), (s_row, s_col-1), (s_row, s_col+1)]
    return [(row, col) for row, col in indices if is_valid(image, row, col)]

def is_valid(image: np.ndarray, row: int, col: int) -> bool:
    return row >= 0 and col >= 0 and row < image.shape[0] and col < image.shape[1]

def get_convexhull_for_candidate(candidate: List[Tuple[int, int]]):
    candidate_np = np.array(candidate)
    return cv2.convexHull(candidate_np)

def is_rectangular(polygon: Polygon) -> bool:    
    polygon_area = polygon.area
    min_rect_area = polygon.minimum_rotated_rectangle.area
   
    if (polygon_area / min_rect_area) > 0.85:
        print(f"polygon area: {polygon_area}")
        print(f"min rect area: {min_rect_area}")
        return True
    return False

def matches_ff_shape(polygon: Polygon, model_perimeter, model_area) -> bool:
    area = polygon.area
    length = polygon.length
    model_ratio = model_area/model_perimeter
    curr_ratio = area/length
    print("curr_ratio: ", curr_ratio)
    print("model_ratio: ", model_ratio)
    ratio_tolerance = 8
    ratio_match = abs(model_ratio - curr_ratio) <= ratio_tolerance
    return ratio_match


def flood_fill(params: Params, start_row: int, start_col: int, visited: Set[Tuple[int, int]]):
    area_tolerance = 300
    area = 0
    candidate = []
    candidate.append([start_row, start_col])
    queue = deque([(start_row, start_col)])

    while queue:
        row, col = queue.popleft()
        visited.add((row, col))
        area += params.pixel_size**2

        if area > params.model_area + area_tolerance:
            return None
        
        for r, c in neighbours(params.image, row,col):
            if (r,c) not in visited and pixel_value_in_range(params.image[r][c], params.min_val, params.top_val):
                candidate.append([r,c])
                visited.add((r,c))
                queue.append((r,c))
    
    if area < params.model_area - area_tolerance:
        return None
    
    print(f"area: {area}")
    return candidate

def detect_candidates(params: Params) -> None:
    visited = set()
    candidates = []
    
    image = params.image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if((row,col) not in visited and pixel_value_in_range(image[row][col], params.min_val, params.top_val)):
                candidate = flood_fill(params, row, col, visited)
                if candidate:
                    candidates.append(candidate)
    return candidates

def detect_football_fields(image: np.ndarray, pixel_size: int, min_val: int, top_val: int, model_length: int, model_width: int):
    accepted_candidates = []
    params = Params(image, pixel_size, min_val, top_val, model_length, model_width)
    candidates = detect_candidates(params)
    print(len(candidates))
    for candidate in candidates:
        convex_hull = get_convexhull_for_candidate(candidate).squeeze()
        polygon = Polygon(convex_hull)
        if is_rectangular(polygon) and matches_ff_shape(polygon, params.model_perimeter, params.model_area):
            bbox = cv2.minAreaRect(convex_hull)
            accepted_candidates.append(bbox)
    return accepted_candidates

def draw_min_area_rect(image, bbox):
    #TODO: Zaimplementować rysowanie [obecnie zwracane dane linijka 140]
    # box = cv2.boxPoints(bbox)
    # box = np.int64(box)
    # cv2.drawContours(image, [box], 0, (0, 0, 255), 2) ???
    pass

def main():
    image_path = "nowa_kopia.tif"
    image = read_spatial_raster(image_path)
    indices = [1,2,3,4,5,6,7,8]
    bands = read_raster_bands(image, indices)
    print(len(bands))
    nir = bands[-1]
    nir_array = read_band_as_array(nir)
    min_val = 900
    top_val = 1700
    pixel_size = 3
    model_length = 62
    model_width = 30
    accepted_candidates = detect_football_fields(nir_array, pixel_size, min_val, top_val, model_length, model_width)
    print(len(accepted_candidates))
    """
    Accepted candidates (900 - 1900):
    [[3788.783  2769.446 ]
    [3801.5706 2734.736 ]
    [3807.5486 2736.9387]
    [3794.761  2771.6487]]

    [[5615.8906 2091.6377]
    [5661.     2078.    ]
    [5662.752  2083.796 ]
    [5617.6426 2097.4336]]

    [[6711.814  6194.59  ]
    [6741.2925 6148.8984]
    [6745.9614 6151.911 ]
    [6716.483  6197.6025]]
    
    """
    normalized_nir = cv2.normalize(nir_array, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    print(normalized_nir.shape)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 800, 800)
    for bbox in accepted_candidates:
        box = cv2.boxPoints(bbox)
        # #switch x and y
        box = np.array([[point[1], point[0]] for point in box])
        box = np.int64(box)
        cv2.drawContours(normalized_nir, [box], 0, (255), 2)
    cv2.imshow("image", normalized_nir)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()