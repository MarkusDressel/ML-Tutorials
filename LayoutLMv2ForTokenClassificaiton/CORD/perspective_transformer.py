
from typing import List, Sequence, Union
import numpy as np

import cv2
import json 
import numpy as np


def keypoint_scale(keypoint: Sequence[float], scale_x: float, scale_y: float)->Sequence[float]:
	"""
	Scales a keypoint by scale_x and scale_y.
    
	Parameters
	----------
	keypoint : Sequence[float]
		A keypoint `(x, y, angle, scale)`.
	scale_x : float
		Scale coefficient x-axis.
	scale_y : float
		Scale coefficient y-axis.

	Returns
	-------
		A keypoint `(x, y, angle, scale)`.
	"""

	x, y, angle, scale = keypoint[:4]
	return x * scale_x, y * scale_y, angle, scale * max(scale_x, scale_y)

def rotation2DMatrixToEulerAngles(matrix: np.ndarray):
    return np.arctan2(matrix[1, 0], matrix[0, 0])

def perspective_keypoint(
    keypoint: Union[List[int], List[float]],
    height: int,
    width: int,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
):
    x, y, angle, scale = keypoint

    keypoint_vector = np.array([x, y], dtype=np.float32).reshape([1, 1, 2])

    x, y = cv2.perspectiveTransform(keypoint_vector, matrix)[0, 0]
    angle += rotation2DMatrixToEulerAngles(matrix[:2, :2])

    scale_x = np.sign(matrix[0, 0]) * np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sign(matrix[1, 1]) * np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    scale *= max(scale_x, scale_y)

    if keep_size:
        scale_x = width / max_width
        scale_y = height / max_height
        return keypoint_scale((x, y, angle, scale), scale_x, scale_y)

    return x, y, angle, scale

def perspective_bbox(
    bbox: Sequence[float],
    height: int,
    width: int,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
):

	x1, y1, x2, y2 = bbox

	points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

	x1, y1, x2, y2 = float("inf"), float("inf"), 0, 0
	for pt in points:
		pt = perspective_keypoint(pt.tolist() + [0, 0], height, width, matrix, max_width, max_height, keep_size)
		x, y = pt[:2]
		x = np.clip(x, 0, width if keep_size else max_width)
		y = np.clip(y, 0, height if keep_size else max_height)
		x1 = min(x1, x)
		x2 = max(x2, x)
		y1 = min(y1, y)
		y2 = max(y2, y)

	x = np.clip([x1, x2], 0, width if keep_size else max_width)
	y = np.clip([y1, y2], 0, height if keep_size else max_height)
	return (x[0], y[0], x[1], y[1])  

def four_point_transform(image, pts, bboxes = None):
	# obtain a consistent order of the points and unpack them
	# individually
	(tl, tr, br, bl) = pts
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(pts, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	adjusted_bboxes = []
	if bboxes:
		rows, cols = image.shape[:2]
		for bbox in bboxes:
			float_bbox = perspective_bbox(bbox,rows, cols, M, maxWidth, maxHeight, False)
			int_bbox = [int(coord) for coord in float_bbox]
			adjusted_bboxes.append(int_bbox)

	# return the warped image
	return warped, adjusted_bboxes