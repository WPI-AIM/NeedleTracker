'''
Created on Oct 1, 2017

@author: Joe Schornak
'''

import time
import numpy as np
import cv2
import subprocess
import socket
import struct
import argparse
import xml.etree.ElementTree as ET
import yaml
from collections import deque
import serial



tree = ET.parse('config.xml')
root = tree.getroot()
ip_address = str(root.find("ip").text)
port = int(root.find("port").text)
index_camera_side = int(root.find("index_camera_side").text)

def main():
    global STATE
    global use_connection
    global use_recorded_video
    global load_video_path

    command = 'v4l2-ctl -d /dev/video' + str(index_camera_side) + ' -c focus_auto=0'
    process2 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    cv2.waitKey(100)
    command = 'v4l2-ctl -d /dev/video' + str(index_camera_side) + ' -c focus_absolute=50'
    process3 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    cv2.waitKey(100)

    # cap_top = cv2.VideoCapture(1)  # Top camera
    cap_side = cv2.VideoCapture(index_camera_side)  # Side camera

    cal_left = Struct(**yaml.load(file('left.yaml','r')))

    mat_left = yaml_to_mat(cal_left.camera_matrix)
    dist_left = yaml_to_mat(cal_left.distortion_coefficients)


    cv2.namedWindow("Camera Side")

    phantom_board_data = np.load("./data/phantom_board.npz")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    board = cv2.aruco.Board_create(objPoints=phantom_board_data['obj_points'], dictionary=dictionary, ids=phantom_board_data['ids'])

    time_start = time.clock()

    while cap_side.isOpened():
        # ret, frame_top = cap_top.read()
        ret, frame_side = cap_side.read()

        if cv2.waitKey(10) == ord('q') or frame_side is None:
            break



        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(image=frame_side, dictionary=dictionary)
        # print(markerIds)
        if markerIds is not None:
            ret, rvec, tvec = cv2.aruco.estimatePoseBoard(corners=markerCorners, ids=markerIds, board=board, cameraMatrix=mat_left, distCoeffs=dist_left)

            if ret:
                rmat, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
                transform_homogeneous = np.concatenate(
                    (np.concatenate((rmat, tvec), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
                print(transform_homogeneous)

                frame_side = cv2.aruco.drawAxis(image=frame_side, cameraMatrix=mat_left, distCoeffs=dist_left, rvec=rvec, tvec=tvec, length=0.03)

        cv2.imshow('Camera Side', frame_side)

    cap_side.release()

    cv2.destroyAllWindows()

# class Triangulator:
#     def __init__(self, P1, P2):
#         self.P1 = P1
#         self.P2 = P2
#
#     def _to_float(self, coords):
#         return (float(coords[0]), float(coords[1]))
#
#     def get_position_3D(self, coords_top, coords_side):
#         pose_3D_homogeneous = cv2.triangulatePoints(self.P1, self.P2,
#                                                     np.array(self._to_float(coords_top)).reshape(2, -1),
#                                                     np.array(self._to_float(coords_side)).reshape(2, -1))
#         return (pose_3D_homogeneous / pose_3D_homogeneous[3])[0:3]
#
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
#
# def draw(img, corners, imgpts):
#     corner = tuple(corners[0].ravel())
#     img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
#     return img
#
# def find_phantom_markers(image):
#     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     segmented = cv2.inRange(image_hsv, np.array([0,0,0]), np.array([180, 255, 80]))
#     cv2.imshow("Seg", segmented)
#     img, contours, hierarchy = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     image_contours = image
#     markers = []
#     if len(contours) > 0:
#         areas = []
#         for i, c in enumerate(contours):
#             area = cv2.contourArea(c)
#             areas.append(area)
#         contours_sorted = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)
#
#         cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 3)
#         cv2.imshow("Contours", image_contours)
#         # print(contours_sorted)
#
#         for contour in contours:
#             M = cv2.moments(contour)
#             if M['m00'] > 0.0:
#                 cx = int(M['m10'] / M['m00'])
#                 cy = int(M['m01'] / M['m00'])
#                 markers.append([[cx, cy]])
#     return np.array(markers, dtype=np.float32)
#
#
def yaml_to_mat(input):
    obj = Struct(**input)
    return np.reshape(np.array(obj.data),(obj.rows,obj.cols))
#
# def draw_target_marker(image, target_coords):
#     output = image.copy()
#     cv2.circle(output, target_coords, 10, (0, 255, 0))
#     return output
#
# def transform_to_robot_coords(input):
#     return np.array([-input[2], input[1], -input[0]])
#
# def make_OIGTL_homogeneous_tform(input_tform):
#     body = struct.pack('!12f',
#                        float(input_tform((0,0))), float(input_tform((1,0))), float(input_tform((2,0))),
#                        float(input_tform((0, 1))), float(input_tform((1, 1))), float(input_tform((2, 1))),
#                        float(input_tform((0, 2))), float(input_tform((1, 2))), float(input_tform((2, 2))),
#                        float(input_tform((0, 3))), float(input_tform((1, 3))), float(input_tform((2, 3))))
#     bodysize = 48
#     return struct.pack('!H12s20sIIQQ', 1, str('TRANSFORM'), str('SIMULATOR'), int(time.time()), 0, bodysize, 0) + body


if __name__ == '__main__':
    main()
