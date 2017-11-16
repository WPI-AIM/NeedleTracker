import numpy as np
import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)


length_phantom = 0.120
height_phantom = 0.058

width_marker = 0.0035

markers_x = 2
markers_y = 2

ids = np.array([1])

# points_corners = np.array([[[0, 0, 0], [100, 0, 0],[100, 100, 0],[0, 100, 0]],
#                            [[200, 0, 0], [300, 0, 0], [300, 100, 0], [200, 100, 0]]],dtype=np.float32)

points_corners = np.array([[[0, 0, 0], [50, 0, 0],[50, -50, 0],[0, -50, 0]]],dtype=np.float32)

board = cv2.aruco.Board_create(objPoints=points_corners, dictionary=dictionary, ids=ids, )
# board = cv2.aruco.GridBoard_create(2, 2, 50, 50, dictionary)


img = cv2.aruco.drawPlanarBoard(board, (300,300))

# img = board.draw((800, 800), 0, 0)
cv2.imshow("Board", img)
cv2.waitKey(0)