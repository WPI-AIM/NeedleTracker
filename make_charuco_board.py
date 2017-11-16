import cv2

dpi=200.0
square_width = 0.007
marker_width = square_width*0.5
squares_wide= 7.0
squares_high = 9.0
in_per_m = 1/2.54*100

pixel_width = int(squares_wide*square_width*in_per_m*dpi)
pixel_height = int(squares_high*square_width*in_per_m*dpi)
print("width",pixel_width, "height",pixel_height)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

board = cv2.aruco.CharucoBoard_create(int(squares_wide), int(squares_high), square_width, marker_width, dictionary)
img = board.draw((pixel_width, pixel_height), 0, 0)
cv2.imshow("Board", img)
cv2.imwrite("/home/jgschornak/board_7mm.png", img)
cv2.waitKey(0)