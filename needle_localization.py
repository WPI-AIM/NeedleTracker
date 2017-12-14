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
parser.add_argument('--use_connection', action='store_true',
                    help='Attempt to connect to the robot control computer.')
parser.add_argument('--use_recorded_video', action='store_true',
                    help='Load and process video from file, instead of trying to get live video from webcams.')
parser.add_argument('--load_video_path', type=str, nargs=1, default='./data/test',
                    help='Path for video to load if --use_recorded_video is specified.')
parser.add_argument('--save_video', action='store_true',
                    help='Save input and output video streams for diagnostic or archival purposes.')
parser.add_argument('--use_target_segmentation', action='store_true',
                    help='Track the target as the largest blob of the color specified in the config file. Default uses manually-picked point.')
parser.add_argument('--use_arduino', action='store_true',
                    help='Trigger an Arduino over USB, to facilitate data logging with external tracking hardware.')
parser.add_argument('--no_registration', action='store_true',
                    help='Calculate 3D coordinates in the frame of the stereo system.')
parser.add_argument('--refraction_compensation', action='store_true',
                    help='Correct for refractive effects caused by phantom medium.')
args = parser.parse_args()
globals().update(vars(args))

TARGET_TOP_A = (int(258), int(246))
TARGET_SIDE_A = (int(261), int(230))

TARGET_TOP_B = (int(0), int(0))
TARGET_SIDE_B = (int(0), int(0))

ROI_CENTER_MANUAL_TOP = (0,0)
ROI_CENTER_MANUAL_SIDE = (0,0)
MANUAL_ROI_TOP_SET = False
MANUAL_ROI_SIDE_SET = False


SEND_MESSAGES = False

MAG_THRESHOLD = 10
FRAME_THRESHOLD = 5

TARGET_SET = False

def main():
    global SEND_MESSAGES, STATE, load_video_path, use_connection, use_recorded_video,\
        load_video_path, save_video, use_target_segmentation, use_arduino, no_registration,\
        FRAME_SIZE, TARGET_SET, ROI_CENTER_MANUAL_TOP, ROI_CENTER_MANUAL_SIDE, MANUAL_ROI_TOP_SET, MANUAL_ROI_SIDE_SET



    # Load xml config file. This is for values that possibly need to be changed but are likely to stay the same for many runs.
    tree = ET.parse('config.xml')
    root = tree.getroot()
    ip_address = str(root.find("ip").text)
    port = int(root.find("port").text)
    output_dir = str(root.find("output_dir").text)
    output_prefix = str(root.find("prefix").text)
    hue_motion = int(root.find("hue_motion").text)
    hue_motion_range = int(root.find("hue_motion_range").text)

    target_hue_min = int(root.find("hue_target_min").text)
    target_hue_max = int(root.find("hue_target_max").text)
    target_sat_min = int(root.find("sat_target_min").text)
    target_sat_max = int(root.find("sat_target_max").text)
    target_val_min = int(root.find("val_target_min").text)
    target_val_max = int(root.find("val_target_max").text)

    camera_top_focus_absolute = int(root.find("camera_top_focus_absolute").text)
    camera_top_contrast = int(root.find("camera_top_contrast").text)
    camera_top_brightness = int(root.find("camera_top_brightness").text)

    camera_side_focus_absolute = int(root.find("camera_side_focus_absolute").text)
    camera_side_contrast = int(root.find("camera_side_contrast").text)
    camera_side_brightness = int(root.find("camera_side_brightness").text)

    index_camera_top = int(root.find("index_camera_top").text)
    index_camera_side = int(root.find("index_camera_side").text)
    index_camera_aux = int(root.find("index_camera_aux").text)

    dof_params_top = root.find("dof_top")
    dof_params_side = root.find("dof_side")

    roi_width = int(root.find("roi_width").text)
    roi_height = int(root.find("roi_height").text)

    output_path = output_dir + output_prefix + '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    print(output_path)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if use_connection:
        print('Connecting to ' + ip_address + ' port ' + str(port) + '...')
        s.connect((ip_address, port))

    arduino = None
    if use_arduino:
        arduino = serial.Serial('/dev/ttyACM0', 19200, timeout=.5)

    process4 = subprocess.Popen(["mkdir", "-p", output_path], stdout=subprocess.PIPE)
    cv2.waitKey(100)

    if not use_recorded_video:
        # For both cameras, turn off autofocus and set the same absolute focal depth the one used during calibration.

        process = subprocess.Popen(["v4l2-ctl", "-d", "/dev/video" + str(index_camera_top), "-c", "focus_auto=0"], stdout=subprocess.PIPE)

        process1 = subprocess.Popen(["v4l2-ctl", "-d", "/dev/video" + str(index_camera_side), "-c", "focus_auto=0"], stdout=subprocess.PIPE)

        process2 = subprocess.Popen(["v4l2-ctl", "-d", "/dev/video" + str(index_camera_top), "-c", "focus_absolute=" + str(camera_top_focus_absolute)], stdout=subprocess.PIPE)

        process3 = subprocess.Popen(["v4l2-ctl", "-d", "/dev/video" + str(index_camera_side), "-c", "focus_absolute=" + str(camera_side_focus_absolute)], stdout=subprocess.PIPE)

        process4 = subprocess.Popen(["v4l2-ctl", "-d /dev/video" + str(index_camera_top), "-c", "contrast="+ str(camera_top_contrast), "brightness="+ str(camera_top_brightness)], stdout=subprocess.PIPE)

        process5 = subprocess.Popen(["v4l2-ctl", "-d /dev/video" + str(index_camera_side), "-c", "contrast="+ str(camera_side_contrast), "brightness="+ str(camera_side_brightness)], stdout=subprocess.PIPE)

        cap_top = cv2.VideoCapture(index_camera_top)  # Top camera
        cap_side = cv2.VideoCapture(index_camera_side)  # Side camera
    else:
        # If live video isn't available, use recorded insertion video
        cap_top = cv2.VideoCapture(load_video_path[0] + '/output_top.avi')
        cap_side = cv2.VideoCapture(load_video_path[0] + '/output_side.avi')

    cap_aux = cv2.VideoCapture(index_camera_aux)

    if no_registration:
        transform_camera_to_registration_marker = np.eye(4)
    else:
        transform_camera_to_registration_marker = np.load("./data/transform_registration_marker.npz")['transform_registration_marker']
    print("Reg Marker Tform Loaded:")
    print(transform_camera_to_registration_marker)

    cal_left = Struct(**yaml.load(file('left.yaml','r')))
    cal_right = Struct(**yaml.load(file('right.yaml', 'r')))

    mat_left_obj = Struct(**cal_left.camera_matrix)
    mat_left = np.reshape(np.array(mat_left_obj.data),(mat_left_obj.rows,mat_left_obj.cols))
    dist_left_obj = Struct(**cal_left.distortion_coefficients)
    dist_left = np.reshape(np.array(dist_left_obj.data), (dist_left_obj.rows, dist_left_obj.cols))

    mat_right_obj = Struct(**cal_right.camera_matrix)
    mat_right = np.reshape(np.array(mat_right_obj.data),(mat_right_obj.rows,mat_right_obj.cols))

    # translation_side_to_top = np.array([9.336674963296142e-05, 0.1268878884308696, 0.12432346740907979]).reshape((3,1))
    translation_side_to_top = np.array([-0.00405898148084166, 0.12556883059577384, 0.12740889645198586]).reshape((3,1))

    # rotation_side_to_top = np.array( [0.997701828954437, -0.03188640457921707, -0.0597855977973143,
    #                                   -0.058907963785628806, 0.027788165175257316, -0.9978765803839791,
    #                                   0.03348002842894244, 0.9991051371498414, 0.025845940052435786]).reshape((3,3))

    rotation_side_to_top = np.array([0.9944425578466023, -0.07751605003550875, -0.07124086699640379,
                                     -0.07012885032973017, 0.01697791409053453, -0.9973934503417218,
                                     0.07852352192137778, 0.996846534036221, 0.011447448843998962]).reshape((3, 3))

    transform_side_to_top = make_homogeneous_tform(rotation=rotation_side_to_top, translation=translation_side_to_top)
    transform_top_to_side = np.linalg.inv(transform_side_to_top)
    # print("top to side")
    # print(transform_top_to_side)

    print("side to top")
    print(transform_side_to_top)

    p1_side = np.dot(mat_left, np.concatenate((np.eye(3), np.zeros((3,1))), axis=1))
    # p1_side = np.array([651.2401417664485, 0.0, 313.8361587524414, 0.0, 0.0, 651.2401417664485, 204.06911087036133, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape((3,4))
    p2_top = np.dot(mat_right, transform_side_to_top[0:3,:])
    # p2_top = np.array([651.2401417664485, 0.0, 313.8361587524414, 0.0, 0.0, 651.2401417664485, 204.06911087036133, 116.52837949973384, 0.0, 0.0, 1.0, 0.0]).reshape((3,4))


    top_frames = deque(maxlen=3)
    side_frames = deque(maxlen=3)

    transforms_tip = deque(maxlen=2)

    _, camera_top_last_frame = cap_top.read()
    _, camera_side_last_frame = cap_side.read()

    top_frames.append(camera_top_last_frame)
    side_frames.append(camera_side_last_frame)

    camera_top_height, camera_top_width, channels = camera_top_last_frame.shape
    camera_side_height, camera_side_width, channels = camera_side_last_frame.shape

    FRAME_SIZE = (camera_top_width, camera_top_height)

    camera_top_roi_size = (roi_width, roi_height)
    camera_side_roi_size = (roi_width, roi_height)

    camera_top_roi_center = (int(camera_top_width * 0.8), camera_top_height / 2)
    camera_side_roi_center = (int(camera_side_width * 0.8), camera_side_height / 2)

    delta_last = None
    position_tip_last = None

    trajectory_corrected = []
    trajectory_uncorrected = []

    target_corrected = []

    transforms_registration_marker_to_tip = []
    transforms_registration_marker_to_target = []
    timestamps = []

    top_path = []
    side_path = []

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out_combined = cv2.VideoWriter(
        filename=output_path + '/output_combined.avi',
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=20.0,
        frameSize=(int(camera_top_width * 2), int(camera_top_height * 2)),
        isColor=True)

    out_top = cv2.VideoWriter(
        filename=output_path + '/output_top.avi',
        fourcc=fourcc,
        fps=20.0,
        frameSize=(camera_top_width, camera_top_height),
        isColor=True)

    out_side = cv2.VideoWriter(
        filename=output_path + '/output_side.avi',
        fourcc=fourcc,
        fps=20.0,
        frameSize=(camera_side_width, camera_side_height),
        isColor=True)

    out_flow = cv2.VideoWriter(
        filename=output_path + '/output_flow.avi',
        fourcc=fourcc,
        fps=10.0,
        frameSize=(camera_top_roi_size[0] * 2, camera_top_roi_size[1] * 2),
        isColor=True)

    cv2.namedWindow("Combined")
    cv2.setMouseCallback("Combined", get_coords)
    cv2.moveWindow("Combined", 1, 0)

    camera_top_farneback_parameters = (float(dof_params_top.find("pyr_scale").text),
                                       int(dof_params_top.find("levels").text),
                                       int(dof_params_top.find("winsize").text),
                                       int(dof_params_top.find("iterations").text),
                                       int(dof_params_top.find("poly_n").text),
                                       float(dof_params_top.find("poly_sigma").text),
                                       0)

    camera_side_farneback_parameters = (float(dof_params_side.find("pyr_scale").text),
                                       int(dof_params_side.find("levels").text),
                                       int(dof_params_side.find("winsize").text),
                                       int(dof_params_side.find("iterations").text),
                                       int(dof_params_side.find("poly_n").text),
                                       float(dof_params_side.find("poly_sigma").text),
                                       0)

    tracker_top = tracking.TipTracker(camera_top_farneback_parameters,
                                      camera_top_width,
                                      camera_top_height,
                                      hue_motion,
                                      hue_motion_range,
                                      int(root.find("threshold_mag").text),
                                      camera_top_roi_center,
                                      camera_top_roi_size,
                                      int(root.find("kernel_top").text),
                                      "camera_top",
                                      verbose=False)
    tracker_side = tracking.TipTracker(camera_side_farneback_parameters,
                                       camera_side_width,
                                       camera_side_height,
                                       hue_motion,
                                       hue_motion_range,
                                       int(root.find("threshold_mag").text),
                                       camera_side_roi_center,
                                       camera_side_roi_size,
                                       int(root.find("kernel_side").text),
                                       "camera_side", verbose=False)

    phantom_dims = np.array([0.12, 0.058, 0.058]) # length is actually 0.12675 meters
    phantom_transform = np.load("./data/transform_camera_to_phantom.npz")['transform_camera_to_phantom']
    # phantom_transform[2,3]=0.1
    camera_a_origin = np.array([0,0,0])
    camera_b_origin = transform_side_to_top[0:3,3].ravel()

    compensator_tip = refraction.RefractionModeler(camera_a_origin,
                                                   camera_b_origin,
                                                   phantom_dims,
                                                   phantom_transform,
                                                   float(root.find("index_refraction_phantom").text),
                                                   float(root.find("index_refraction_ambient").text))

    compensator_target = refraction.RefractionModeler(camera_a_origin,
                                                      camera_b_origin,
                                                      phantom_dims,
                                                      phantom_transform,
                                                      float(root.find("index_refraction_phantom").text),
                                                      float(root.find("index_refraction_ambient").text))

    print("Target seg hue range: " + str(target_hue_min) + " to " + str(target_hue_max))
    target_top = tracking.TargetTracker(target_hue_min,
                                        target_hue_max,
                                        target_sat_min,
                                        target_sat_max,
                                        target_val_min,
                                        target_val_max,
                                        None,
                                        TARGET_TOP_A)

    target_side = tracking.TargetTracker(target_hue_min,
                                         target_hue_max,
                                         target_sat_min,
                                         target_sat_max,
                                         target_val_min,
                                         target_val_max,
                                         None,
                                         TARGET_SIDE_A)

    triangulator_tip = tracking.Triangulator(p1_side,
                                             p2_top)

    triangulator_target = tracking.Triangulator(p1_side,
                                                p2_top)

    phantom_board_data = np.load("./data/phantom_board.npz")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    board = cv2.aruco.Board_create(objPoints=phantom_board_data['obj_points'], dictionary=dictionary, ids=phantom_board_data['ids'])
    tracker_phantom = tracking.PhantomTracker(board, dictionary, mat_left, dist_left, dims_phantom=phantom_dims)


    time_start = time.time()

    while cap_top.isOpened():
        try:
            time_last = time.time()

            _, camera_top_current_frame = cap_top.read()
            _, camera_side_current_frame = cap_side.read()

            input = cv2.waitKey(1)
            if input == ord('q') or camera_top_current_frame is None or camera_side_current_frame is None:
                break
            elif input == ord('a'):

                # Toggle target status
                TARGET_SET = not TARGET_SET
            output_string = ""

            times = []

            if arduino is not None:
                arduino.write('1\n')

            # ret, aux_frame = cap_aux.read()
            aux_frame = None

            timestamps.append(time.time() - time_start)

            top_frames.append(camera_top_current_frame)
            side_frames.append(camera_side_current_frame)

            tracker_side.update(side_frames, MANUAL_ROI_SIDE_SET, ROI_CENTER_MANUAL_SIDE)
            tracker_top.update(top_frames, MANUAL_ROI_TOP_SET, ROI_CENTER_MANUAL_TOP)
            MANUAL_ROI_TOP_SET = False
            MANUAL_ROI_SIDE_SET = False

            tracker_phantom.update(camera_side_current_frame)
            # tracker_phantom.get_phantom_corner_image_points()

            time_delta = time.time() - time_last
            time_last = time.time()
            times.append(time_delta)
            output_string += "2D Localization: " + str(time_delta) + "\n"

            if use_target_segmentation:
                target_top.update(camera_top_current_frame)
                target_side.update(camera_side_current_frame)
                cv2.imshow("Target top", target_top.image_masked)
                cv2.imshow("Target side", target_side.image_masked)
            else:
                target_top.target_coords = TARGET_TOP_A
                target_side.target_coords = TARGET_SIDE_A

            camera_top_with_marker = draw_tip_marker(camera_top_current_frame,
                                                     tracker_top.roi_center,
                                                     tracker_top.roi_size,
                                                     tracker_top.position_tip)

            camera_top_with_marker = draw_target_markers(camera_top_with_marker,
                                                         target_top.target_coords,
                                                         TARGET_TOP_B)

            camera_side_with_marker = draw_tip_marker(camera_side_current_frame,
                                                      tracker_side.roi_center,
                                                      tracker_side.roi_size,
                                                      tracker_side.position_tip)

            camera_side_with_marker = draw_target_markers(camera_side_with_marker,
                                                          target_side.target_coords,
                                                          TARGET_SIDE_B)

            position_tip = triangulator_tip.get_position_3D(tracker_top.position_tip,
                                                            tracker_side.position_tip)

            position_target = triangulator_target.get_position_3D(target_top.target_coords,
                                                                  target_side.target_coords)

            # position_target_second = triangulator_target.get_position_3D(TARGET_TOP_B,
            #                                                              TARGET_SIDE_B)

            success_compensation_tip, position_tip_corrected_list = compensator_tip.solve_real_point_from_refracted(np.ravel(position_tip))
            success_compensation_target, position_target_corrected_list = compensator_target.solve_real_point_from_refracted(np.ravel(position_target))
            position_tip_corrected = position_tip_corrected_list

            position_target_corrected = position_target
            if success_compensation_target:
                position_target_corrected = np.array([position_target_corrected_list[0], position_target_corrected_list[1], position_target_corrected_list[2]]).reshape((3,1))
                output_string += "\n"
            else:
                output_string += "TARGET REFRACTION COMPENSATION FAILED!\n"

            # Don't claim that automatically-generated target poses are accurate if they fail refraction compensation
            if use_target_segmentation:
                if success_compensation_target:
                    TARGET_SET = True
                else:
                    TARGET_SET = False

            time_delta = time.time() - time_last
            time_last = time.time()
            times.append(time_delta)
            output_string += "Triangulation/Refraction: " + str(time_delta) + "\n"

            # transform_camera_to_target_uncorrected = make_homogeneous_tform(translation=position_target)
            transform_camera_to_target = make_homogeneous_tform(translation=position_target_corrected)

            transform_registration_marker_to_camera = np.linalg.inv(transform_camera_to_registration_marker)

            transform_registration_marker_to_target = np.dot(transform_registration_marker_to_camera, transform_camera_to_target)
            transform_registration_marker_to_target[0:3,0:3] = np.eye(3)

            transforms_registration_marker_to_target.append(transform_registration_marker_to_target)


            output_string += "Reg Marker to Target\n" + str(transform_registration_marker_to_target) + "\n"

            if use_connection:
                s.send(compose_OpenIGTLink_transform(transform_registration_marker_to_target))
                if TARGET_SET:
                    s.send(compose_OpenIGTLink_status(1, 0, "TARGET_SET"))
                else:
                    s.send(compose_OpenIGTLink_status(13, 0, "NO_TARGET_SET"))

            transform_registration_marker_to_tip = np.array([[0, 0, 1, 0],
                                                             [0, 0, 0, 0],
                                                             [0, 0, 0, 0],
                                                             [0, 0, 0, 0]])
            if not success_compensation_tip:
                output_string += "TIP REFRACTION COMPENSATION FAILED!\n"
            else:
                output_string += "\n"

            if not np.array_equal(position_tip_corrected, position_tip_last) and success_compensation_tip:
                delta = position_tip_corrected - position_target_corrected
                rotation_tip = np.array([[0.99, 0, 0.1],
                                         [0.01, 0.99, 0],
                                         [0, 0.01, 0.99]])
                if len(transforms_tip) is not 0:
                    direction_motion = normalize(
                        position_tip_corrected.reshape((3, 1)) - transforms_tip[-1][0:3, 3].reshape((3, 1)))
                    axis_y = np.array([0, 1, 0]).reshape((1,3))
                    axis_z = normalize(np.cross(direction_motion.reshape((1,3)), axis_y).reshape((1,3)))
                    axis_y = normalize(np.cross(axis_z.reshape((1,3)), direction_motion.reshape((1,3))))
                    rotation_tip = np.concatenate((direction_motion.reshape((3, 1)),
                                                   axis_y.reshape((3, 1)),
                                                   axis_z.reshape((3, 1))),
                                                  axis=1)

                transform_camera_to_tip = make_homogeneous_tform(rotation=rotation_tip, translation=position_tip_corrected)

                transforms_tip.append(transform_camera_to_tip)

                transform_registration_marker_to_tip = np.dot(np.linalg.inv(transform_camera_to_registration_marker), transform_camera_to_tip)

                output_string += "Reg Marker to Tip\n" + str(transform_registration_marker_to_tip) + "\n"

                position_tip_time = np.concatenate(([[time.time()]], position_tip_corrected.reshape((3,1))))
                position_tip_uncorrected_time = np.concatenate(([[time.time()]], position_tip.reshape((3, 1))))

                trajectory_corrected.append(position_tip_time)
                target_corrected.append(position_target_corrected)
                trajectory_uncorrected.append(position_tip_uncorrected_time)
                transforms_registration_marker_to_tip.append(transform_registration_marker_to_tip)
            else:
                trajectory_corrected.append(np.array([time.time(), 0, 0, 0]).reshape((4,1)))
                target_corrected.append(position_target_corrected)
                trajectory_uncorrected.append(np.array([time.time(), 0, 0, 0]).reshape((4,1)))
                transforms_registration_marker_to_tip.append(np.eye(4))

            top_path.append(tracker_top.position_tip)
            side_path.append(tracker_side.position_tip)

            if use_connection:
                s.send(compose_OpenIGTLink_transform(transform_registration_marker_to_tip))

            camera_top_with_marker = draw_tip_path(camera_top_with_marker,
                                                   top_path)
            camera_side_with_marker = draw_tip_path(camera_side_with_marker,
                                                    side_path)

            # camera_side_with_marker = tracker_phantom.draw_phantom_axes(camera_side_with_marker)

            # camera_side_with_marker = tracker_phantom.draw_phantom_corners(camera_side_with_marker)
            # tracker_phantom.get_phantom_mask(camera_side_with_marker.shape)

            time_delta = time.time() - time_last
            time_last = time.time()
            times.append(time_delta)
            # print("Comms: " + str(time_delta))
            output_string += "Comms: " + str(time_delta) + "\n"

            font = cv2.FONT_HERSHEY_DUPLEX
            text_color = (0, 255, 0)
            data_frame = np.zeros_like(camera_top_with_marker)

            cv2.putText(camera_top_with_marker, "Top", (5,20), font, 0.5, text_color)
            cv2.putText(camera_side_with_marker, "Side", (5,20), font, 0.5, text_color)

            time_delta = time.time() - time_last
            time_last = time.time()
            times.append(time_delta)
            output_string += "Total: " + str(sum(times)) + "\n"

            if aux_frame is not None:
                combined2 = np.concatenate((data_frame, aux_frame), axis=0)
            else:
                combined2 = np.concatenate((data_frame, np.zeros_like(data_frame)), axis=0)

            y0, dy = 50, 25
            for i, line in enumerate(output_string.split('\n')):
                y = y0 + i * dy
                cv2.putText(combined2, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color)

            out_top.write(camera_top_current_frame)
            out_side.write(camera_side_current_frame)

            combined1 = np.concatenate((camera_top_with_marker, camera_side_with_marker), axis=0)
            combined = np.array(np.concatenate((combined1, combined2), axis=1), dtype=np.uint8)

            combined_flow = np.array(np.concatenate((tracker_top.flow_diagnostic, tracker_side.flow_diagnostic), axis=1), dtype=np.uint8)
            cv2.imshow('Combined', combined)
            cv2.imshow('Flow', combined_flow)
            out_combined.write(combined)
            out_flow.write(combined_flow)

            delta_last = delta
            position_tip_last = position_tip_corrected

            time_delta = time.time() - time_last
            times.append(time_delta)

            output_string += "Diagnostics: " + str(time_delta) + "\n"
            print(output_string)
        except socket.error, e:
            print "Error: %s" % e
            break

    if s is not None:
        print("Closing connection")
        s.close()

    cap_top.release()
    cap_side.release()

    out_combined.release()
    out_top.release()
    out_side.release()
    out_flow.release()

    cv2.destroyAllWindows()

    trajectory_array_corrected = np.array(trajectory_corrected)
    target_array = np.array(target_corrected)

    # print(trajectory_array_corrected.shape)
    # print(target_array.shape)

    np.savetxt(output_path + "/trajectory2d.csv", np.concatenate((top_path, side_path), axis=1), delimiter=",")
    np.savez_compressed(output_path + "/trajectory.npz",
                        trajectory_tip=trajectory_array_corrected,
                        trajectory_tip_uncorrected=np.array(trajectory_uncorrected),
                        top_path=np.array(top_path),
                        side_path=np.array(side_path),
                        transforms_registration_marker_to_tip=np.array(transforms_registration_marker_to_tip),
                        transforms_registration_marker_to_target=np.array(transforms_registration_marker_to_target),
                        timestamps=np.array(timestamps))

    # combined_array =np.concatenate((trajectory_array_corrected, target_array),1)
    print(np.array(timestamps).reshape((-1,1)).shape)
    print(np.array(transforms_registration_marker_to_tip)[:,0:3,3].shape)
    print(np.array(transforms_registration_marker_to_target)[:,0:3,3].shape)
    np.savetxt(output_path + "/trajectory.csv", np.concatenate((np.array(timestamps).reshape((-1,1)), np.array(transforms_registration_marker_to_tip)[:,0:3,3], np.array(transforms_registration_marker_to_target)[:,0:3,3]), axis=1), delimiter=",")

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# class plot3dClass(object):
#     def __init__(self, systemSideLength, lowerCutoffLength ):
#         self.systemSideLength = systemSideLength
#         self.lowerCutoffLength = lowerCutoffLength
#
#         self.fig = plt.figure()
#         self.ax = self.fig.add_subplot(111, projection='3d')
#
#         # rng = np.arange(0, self.systemSideLength, self.lowerCutoffLength)
#         # self.X, self.Y = np.meshgrid(rng, rng)
#
#         self.ax.w_zaxis.set_major_locator(LinearLocator(10))
#         self.ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))
#
#         # heightR = np.zeros(self.X.shape)
#         # self.surf = self.ax.plot_surface(
#         #     self.X, self.Y, heightR, rstride=1, cstride=1,
#         #     cmap=cm.jet, linewidth=0, antialiased=False)
#         # plt.draw() maybe you want to see this frame?
#
#     def drawNow(self, point):
#
#         # self.surf.remove()
#         # self.surf = self.ax.plot_surface(
#         #     self.X, self.Y, heightR, rstride=1, cstride=1,
#         #     cmap=cm.jet, linewidth=0, antialiased=False)
#
#         r = [-0.05, 0.05]
#         for s, e in combinations(np.array(list(product(r, r, r))), 2):
#             if np.sum(np.abs(s - e)) == r[1] - r[0]:
#                 self.ax.plot3D(*zip(s, e), color="b")
#         # print(point)
#         self.ax.scatter(point[0], point[1], point[2], color="r")
#
#         plt.draw()  # redraw the canvas
#
#         self.fig.canvas.flush_events()
#         # time.sleep(1)

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

def draw_target_markers(image, target_coords_a, target_coords_b):
    output = image.copy()
    cv2.circle(output, target_coords_a, 10, (0, 255, 0))
    cv2.circle(output, target_coords_b, 10, (255, 0, 255))
    return output

def get_coords(event, x, y, flags, param):
    global TARGET_TOP_A, TARGET_SIDE_A, FRAME_SIZE, ROI_CENTER_MANUAL_TOP, ROI_CENTER_MANUAL_SIDE, MANUAL_ROI_TOP_SET, MANUAL_ROI_SIDE_SET
    if x < FRAME_SIZE[0]:
        if y < FRAME_SIZE[1]:
            # click in top image
            if event == cv2.EVENT_LBUTTONDOWN:
                TARGET_TOP_A = x, y
            elif event == cv2.EVENT_MBUTTONDOWN:
                ROI_CENTER_MANUAL_TOP = x, y
                MANUAL_ROI_TOP_SET = True
        else:
            # click in bottom image
            if event == cv2.EVENT_LBUTTONDOWN:
                TARGET_SIDE_A = x, y-FRAME_SIZE[1]
            elif event == cv2.EVENT_MBUTTONDOWN:
                ROI_CENTER_MANUAL_SIDE = x, y-FRAME_SIZE[1]
                MANUAL_ROI_SIDE_SET = True

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

def compose_OpenIGTLink_transform(input_tform):
    body = struct.pack('!12f',
                       float(input_tform[0, 0]), float(input_tform[1, 0]), float(input_tform[2, 0]),
                       float(input_tform[0, 1]), float(input_tform[1, 1]), float(input_tform[2, 1]),
                       float(input_tform[0, 2]), float(input_tform[1, 2]), float(input_tform[2, 2]),
                       float(input_tform[0, 3]), float(input_tform[1, 3]), float(input_tform[2, 3]))
    bodysize = 48
    return struct.pack('!H12s20sIIQQ', 1, str('TRANSFORM'), str('SIMULATOR'), int(time.time()), 0, bodysize, 0) + body

# def compose_OpenIGTLink_string(input_string):
#     body = struct.pack('!4s', input_string)
#     print(body)
#     bodysize = 4
#     result = struct.pack('!H12s20sIIQQ', 1, str('STRING'), str('SIMULATOR'), int(time.time()), 0, bodysize, 0) + body
#     print(result)
#     return result

def compose_OpenIGTLink_status(input_status, subcode, message):
    body = struct.pack('!HQ20s', input_status, subcode, message)
    bodysize = 30
    return struct.pack('!H12s20sIIQQ', 1, str('STATUS'), str('SIMULATOR'), int(time.time()), 0, bodysize, 0) + body

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
