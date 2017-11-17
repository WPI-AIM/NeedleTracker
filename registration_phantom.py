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

    transform_homogeneous = np.eye(4)

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

    print("Saving...")
    print(transform_homogeneous)
    np.savez_compressed("./data/transfer_camera_to_phantom.npz", transform_camera_to_phantom=transform_homogeneous)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def yaml_to_mat(input):
    obj = Struct(**input)
    return np.reshape(np.array(obj.data),(obj.rows,obj.cols))

if __name__ == '__main__':
    main()
