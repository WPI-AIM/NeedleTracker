import numpy as np
import cv2

def make_corner_points(top_left_corner, width_marker):
    return np.array([[top_left_corner, top_left_corner + np.array([width_marker, 0, 0]), top_left_corner + np.array([width_marker, -width_marker, 0]), top_left_corner + np.array([0, -width_marker, 0])]], dtype=np.float32)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)


length_phantom = 0.120
height_phantom = 0.058 # more like 0.05785
px_per_m = 200*(1/2.54)*100

length_phantom_px = length_phantom*px_per_m
height_phantom_px = height_phantom*px_per_m
print(length_phantom_px, height_phantom_px)

width_marker = 0.0035
width_marker_px = width_marker*px_per_m

markers_x_count = 12
markers_y_count = 6

# ids = np.array([1])

# points_corners = np.array([[[0, 0, 0], [100, 0, 0],[100, 100, 0],[0, 100, 0]],
#                            [[200, 0, 0], [300, 0, 0], [300, 100, 0], [200, 100, 0]]],dtype=np.float32)

top_left = [0, 0, 0]
top_right = [length_phantom_px - width_marker_px, 0, 0]
bottom_left = [0, height_phantom_px - width_marker_px, 0]
bottom_right = [length_phantom_px - width_marker_px, height_phantom_px - width_marker_px, 0]

phantom_center_to_grid_origin = np.array([-length_phantom/2, -height_phantom/2, height_phantom/2])

points_corners_list = []
for coord_x in np.linspace(0, length_phantom - width_marker, markers_x_count, endpoint=True):
    print(coord_x)
    points_corners_list.append(make_corner_points(phantom_center_to_grid_origin + np.array([coord_x, 0, 0]), width_marker))
    points_corners_list.append(make_corner_points(phantom_center_to_grid_origin + np.array([coord_x, height_phantom - width_marker, 0]), width_marker))

for coord_y in np.linspace(height_phantom*0.2, height_phantom*0.75, 4, endpoint=True):
    points_corners_list.append(make_corner_points(phantom_center_to_grid_origin + np.array([0, coord_y, 0]), width_marker))


points_corners = np.concatenate(tuple(points_corners_list))
# points_corners = np.array([[[0, 0, 0], [50, 0, 0],[50, -50, 0],[0, -50, 0]]],dtype=np.float32)
# points_corners = np.concatenate((make_corner_points(np.array([0,0,0]), width_marker),make_corner_points(np.array([0.01,0,0]), width_marker)))
# print(points_corners.shape)

ids = np.array(range(0,points_corners.shape[0]))

# obj points need to be in meters
board = cv2.aruco.Board_create(objPoints=points_corners, dictionary=dictionary, ids=ids)
# board = cv2.aruco.GridBoard_create(2, 2, 50, 50, dictionary)

# size board by px-to-m conversion
img = cv2.aruco.drawPlanarBoard(board, (int(length_phantom_px),int(height_phantom_px)))

# img = board.draw((800, 800), 0, 0)
# cv2.imshow("Board", img)
# cv2.waitKey(0)
# cv2.imwrite("./data/phantom_board.png", img)

np.savez_compressed("./data/phantom_board.npz", obj_points = points_corners, ids = ids)