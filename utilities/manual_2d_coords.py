'''
needle_localization.py
Main script to handle 3D needle tracking in a transparent phantom with highly-converging stereo cameras.
Created on Feb 7, 2017

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
from collections import deque
import yaml
import refraction
import tracking
import serial
import matplotlib

matplotlib.interactive(True)

# Parse command line arguments. These are options for things likely to change between runs.
parser = argparse.ArgumentParser(description='Do 3D localization of a needle tip using dense optical flow.')
parser.add_argument('--load_video_path', type=str, nargs=1, default='./data/test',
                    help='Path for video to load if --use_recorded_video is specified.')
args = parser.parse_args()
globals().update(vars(args))

TARGET_TOP_A = (int(258), int(246))
TARGET_SIDE_A = (int(261), int(230))


def main():
    global load_video_path, FRAME_SIZE, TARGET_TOP_A, TARGET_SIDE_A

    output_path = '/home/jgschornak/NeedleLocalization/data/ground_truth_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    print(output_path)

    process4 = subprocess.Popen(["mkdir", "-p", output_path], stdout=subprocess.PIPE)
    cv2.waitKey(100)
    # process4.kill()

    print(load_video_path)
    # If live video isn't available, use recorded insertion video
    cap_top = cv2.VideoCapture('/home/jgschornak/NeedleLocalization/data/validation_2d_2017_11_28_12_00_46/output_top.avi')
    cap_side = cv2.VideoCapture('/home/jgschornak/NeedleLocalization/data/validation_2d_2017_11_28_12_00_46/output_side.avi')

    cv2.waitKey(500)

    top_frames = deque(maxlen=3)
    side_frames = deque(maxlen=3)


    _, camera_top_last_frame = cap_top.read()
    _, camera_side_last_frame = cap_side.read()

    top_frames.append(camera_top_last_frame)
    side_frames.append(camera_side_last_frame)

    camera_top_height, camera_top_width, channels = camera_top_last_frame.shape
    camera_side_height, camera_side_width, channels = camera_side_last_frame.shape

    FRAME_SIZE = (camera_top_width, camera_top_height)

    top_path = []
    side_path = []

    cv2.namedWindow("Combined")
    cv2.setMouseCallback("Combined", get_coords)

    stop = False


    while cap_top.isOpened():
        _, camera_top_current_frame = cap_top.read()
        _, camera_side_current_frame = cap_side.read()

        if camera_top_current_frame is None or camera_side_current_frame is None or stop:
            break

        TARGET_TOP_A = (0, 0)
        TARGET_SIDE_A = (0, 0)
        print("New frame")
        while True:
            value = cv2.waitKey(1)
            if value is ord('a'):
                break
            elif value is ord('q'):
                stop = True
                break

            camera_top_with_marker = draw_target_markers(camera_top_current_frame,
                                                         TARGET_TOP_A)

            camera_side_with_marker = draw_target_markers(camera_side_current_frame,
                                                          TARGET_SIDE_A)

            combined = np.concatenate((camera_top_with_marker, camera_side_with_marker), axis=0)
            cv2.imshow("Combined", combined)
        top_path.append(TARGET_TOP_A)
        side_path.append(TARGET_SIDE_A)

    cap_top.release()
    cap_side.release()

    cv2.destroyAllWindows()

    paths = np.concatenate((top_path, side_path), axis=1)
    print(paths)

    np.savetxt(output_path + '/trajectory2d_canonical.csv', paths, delimiter=",")
    cv2.waitKey(500)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def draw_tip_marker(image, roi_center, roi_size, tip_position):
    line_length = 50
    output = image.copy()
    cv2.circle(output, tip_position, 10, (0, 0, 255))
    cv2.rectangle(output, (roi_center[0] - roi_size[0] / 2, roi_center[1] - roi_size[1] / 2),
                  (roi_center[0] + roi_size[0] / 2, roi_center[1] + roi_size[1] / 2), (0, 0, 255), 1)
    # cv2.line(output, tip_position, (int(tip_position[0] - line_length*math.cos(tip_heading)),
    # int(tip_position[1] - line_length*math.sin(tip_heading))), (0,255,0))
    return output

def draw_tip_path(image, path):
    output = image.copy()
    for point in path:
        cv2.circle(output, (int(point[0]), int(point[1])), 7, (80, 127, 255))
    return output

def draw_target_markers(image, target_coords_a):
    output = image.copy()
    cv2.circle(output, target_coords_a, 10, (0, 255, 0))
    return output

def get_coords(event, x, y, flags, param):
    global TARGET_TOP_A, TARGET_TOP_B, TARGET_SIDE_A, TARGET_SIDE_B, FRAME_SIZE
    if x < FRAME_SIZE[0]:
        if y < FRAME_SIZE[1]:
            # click in top image
            if event == cv2.EVENT_LBUTTONDOWN:
                print("Received top image click")
                TARGET_TOP_A = x, y
        else:
            # click in bottom image
            if event == cv2.EVENT_LBUTTONDOWN:
                print("Received side image click")
                TARGET_SIDE_A = x, y-FRAME_SIZE[1]

def transform_to_robot_coords(input):
    return np.array([-input[2], input[1], -input[0]])

def normalize(input):
    return np.array(input / np.linalg.norm(input))

def is_within_bounds(input):
    x_bound = (-60, 80)
    y_bound = (-40, 40)
    z_bound = (70, 210)

    if input[0] >= (x_bound[0] and input[0] <= x_bound[1] and input[1] >= y_bound[0] and input[1] <= y_bound[1]
                    and input[2] >= z_bound[0] and input[2] <= z_bound[1]):
        print('Within bounds!')
        return True
    else:
        return False

def make_homogeneous_tform(rotation=np.eye(3), translation=np.zeros((3,1))):
    homogeneous = np.eye(4)
    homogeneous[0:3, 0:3] = rotation
    homogeneous[0:3, 3] = translation.reshape((3,1))[:,0]
    return homogeneous

def compose_OpenIGTLink_message(input_tform):
    body = struct.pack('!12f',
                       float(input_tform[0, 0]), float(input_tform[1, 0]), float(input_tform[2, 0]),
                       float(input_tform[0, 1]), float(input_tform[1, 1]), float(input_tform[2, 1]),
                       float(input_tform[0, 2]), float(input_tform[1, 2]), float(input_tform[2, 2]),
                       float(input_tform[0, 3]), float(input_tform[1, 3]), float(input_tform[2, 3]))
    bodysize = 48
    return struct.pack('!H12s20sIIQQ', 1, str('TRANSFORM'), str('SIMULATOR'), int(time.time()), 0, bodysize, 0) + body

def drawlines(img1, line):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, channels = img1.shape
    line = line[0][0]
    # print(line)
    color = (255, 0, 0)
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
    img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
    return img1

def make_data_string(data):
    return '%0.3g, %0.3g, %0.3g' % (data.ravel()[0], data.ravel()[1], data.ravel()[2])

if __name__ == '__main__':
    main()
