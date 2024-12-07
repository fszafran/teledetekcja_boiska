import numpy as np
from collections import deque
from typing import Set, Tuple, List
import cv2
from shapely.geometry import Polygon
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
    area_tolerance = 200
    area = 0
    candidate = set()
    candidate.add((start_row, start_col))
    queue = deque([(start_row, start_col)])

    while queue:
        row, col = queue.popleft()
        visited.add((row, col))
        area += params.pixel_size**2

        if area > params.model_area + area_tolerance:
            return None
        
        for r, c in neighbours(params.image, row,col):
            if (r,c) not in visited and pixel_value_in_range(params.image[r][c], params.min_val, params.top_val):
                candidate.add((r,c))
                visited.add((r,c))
                queue.append((r,c))
    
    if area < params.model_area - area_tolerance:
        return None
    
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
    polygon = Polygon(convex_hull)
    polygon_area = polygon.area
    min_rect_area = polygon.minimum_rotated_rectangle.area
    
    if (polygon_area / min_rect_area) > 0.7 or (min_rect_area / polygon_area) > 0.7:
        return True
    return False

def detect_football_fields(image: np.ndarray, pixel_size: int, min_val: int, top_val: int, model_length: int, model_width: int):
    accepted_candidates = []
    params = Params(image, pixel_size, min_val, top_val, model_length, model_width)
    candidates = detect_candidates(params)
    for candidate in candidates:
        convex_hull = get_convexhull_for_candidate(candidate)
        if is_rectangular(convex_hull):
            bbox = cv2.minAreaRect(convex_hull)
            accepted_candidates.append(bbox)
    return accepted_candidates