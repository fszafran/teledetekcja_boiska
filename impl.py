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
        self.model_area = model_length * model_width
        self.model_length = model_length
        self.model_width = model_width

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

def pixel_value_in_range(pixel_value: int , min_value: int, top_value: int) -> bool:
    return min_value <= pixel_value <= top_value

def flood_fill(params: Params, start_row: int, start_col: int, visited: Set[Tuple[int, int]]):
    area_tolerance = 10
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

def neighbours(image: np.ndarray, s_row: int , s_col: int) -> List[Tuple[int, int]]:
    indices = [(s_row-1, s_col), (s_row+1, s_col), (s_row, s_col-1), (s_row, s_col+1)]
    return [(row, col) for row, col in indices if is_valid(image, row, col)]

def is_valid(image: np.ndarray, row: int, col: int) -> bool:
    return row >= 0 and col >= 0 and row < image.shape[0] and col < image.shape[1]

def get_convexhull_for_candidate(candidate: List[Tuple[int, int]]):
    candidate_np = np.array(candidate)
    return cv2.convexHull(candidate_np)

def is_rectangular(convex_hull: np.ndarray) -> bool:
    convex_hull = convex_hull.squeeze()
    if len(convex_hull) < 4:
        return False
    
    polygon = Polygon(convex_hull)
    polygon_area = polygon.area
    min_rect_area = polygon.minimum_rotated_rectangle.area
   
    if (polygon_area / min_rect_area) > 0.85:
        print(f"polygon area: {polygon_area}")
        print(f"min rect area: {min_rect_area}")
        return True
    return False

def detect_football_fields(image: np.ndarray, pixel_size: int, min_val: int, top_val: int, model_length: int, model_width: int):
    accepted_candidates = []
    params = Params(image, pixel_size, min_val, top_val, model_length, model_width)
    candidates = detect_candidates(params)
    print(len(candidates))
    for candidate in candidates:
        convex_hull = get_convexhull_for_candidate(candidate)
        if is_rectangular(convex_hull):
            bbox = cv2.minAreaRect(convex_hull)
            accepted_candidates.append(bbox)
    return accepted_candidates

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

def main():
    image_path = "copy_im.tif"
    image = read_spatial_raster(image_path)
    indices = [1,2,3,4,5,6,7,8]
    bands = read_raster_bands(image, indices)
    print(len(bands))
    nir = bands[-1]
    nir_array = read_band_as_array(nir)
    min_val = 900
    top_val = 1900
    pixel_size = 3
    model_length = 62
    model_width = 30
    accepted_candidates = detect_football_fields(nir_array, pixel_size, min_val, top_val, model_length, model_width)
    print(len(accepted_candidates))
    # camdidate = {(258, 2122), (289, 2100), (299, 2077), (275, 2104), (304, 2085), (252, 2120), (258, 2124), (271, 2109), (273, 2106), (276, 2105), (300, 2078), (254, 2120), (259, 2119), (287, 2080), (301, 2079), (253, 2121), (302, 2080), (273, 2108), (274, 2107), (257, 2121), (285, 2082), (263, 2116), (286, 2081), (292, 2097), (256, 2119), (255, 2121), (283, 2103), (284, 2102), (296, 2076), (277, 2105), (259, 2121), (261, 2118), (305, 2088), (262, 2117), (290, 2099), (267, 2113), (269, 2110), (296, 2094), (306, 2086), (289, 2079), (287, 2100), (302, 2082), (294, 2078), (297, 2077), (302, 2091), (257, 2123), (279, 2105), (303, 2090), (260, 2119), (265, 2115), (294, 2096), (291, 2079), (295, 2095), (285, 2102), (286, 2101), (304, 2082), (304, 2091), (306, 2088), (261, 2120), (305, 2090), (262, 2119), (293, 2097), (298, 2093), (302, 2084), (288, 2101), (303, 2083), (302, 2093), (303, 2092), (260, 2121), (304, 2084), (298, 2077), (304, 2093), (306, 2090), (305, 2092), (262, 2121), (283, 2082), (300, 2077), (278, 2104), (275, 2105), (304, 2086), (305, 2085), (253, 2120), (280, 2104), (266, 2114), (268, 2111), (271, 2110), (273, 2107), (272, 2109), (274, 2106), (257, 2120), (301, 2081), (258, 2119), (302, 2079), (263, 2115), (292, 2078), (292, 2096), (300, 2079), (285, 2081), (255, 2120), (297, 2095), (299, 2092), (256, 2121), (259, 2120), (261, 2117), (305, 2087), (262, 2116), (290, 2098), (288, 2080), (287, 2081), (282, 2103), (302, 2081), (301, 2083), (294, 2077), (257, 2122), (301, 2092), (258, 2121), (260, 2118), (289, 2099), (292, 2098), (294, 2095), (281, 2104), (284, 2103), (293, 2078), (296, 2077), (261, 2119), (305, 2089), (262, 2118), (291, 2099), (293, 2096), (302, 2083), (303, 2082), (257, 2124), (303, 2091), (258, 2123), (260, 2120), (276, 2104), (295, 2096), (299, 2078), (305, 2091), (251, 2120), (262, 2120), (301, 2078), (303, 2084), (303, 2093), (254, 2121), (283, 2102), (305, 2084), (277, 2104), (266, 2113), (301, 2080), (267, 2112), (272, 2108), (274, 2105), (257, 2119), (284, 2082), (273, 2109), (297, 2076), (278, 2105), (279, 2104), (264, 2115), (265, 2114), (290, 2079), (270, 2110), (297, 2094), (256, 2120), (285, 2101), (305, 2086), (280, 2105), (304, 2090), (306, 2087), (268, 2112), (292, 2079), (269, 2111), (296, 2095), (289, 2080), (301, 2091), (258, 2120), (260, 2117), (287, 2101), (288, 2100), (302, 2092), (281, 2103), (295, 2077), (299, 2093), (256, 2122), (300, 2092), (286, 2102), (291, 2098), (304, 2083), (298, 2076), (304, 2092), (306, 2089), (261, 2121), (301, 2084), (303, 2081), (298, 2094), (301, 2093)}
    # candidate_list_of_lists = [list(point) for point in camdidate]
    # camdidate = np.array(candidate_list_of_lists)
    # ch = cv2.convexHull(camdidate)
    # convex_hull = ch.squeeze()
    # print(convex_hull)
    # print(is_rectangular(convex_hull))
if __name__ == "__main__":
    main()