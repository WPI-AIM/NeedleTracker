'''
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib

from itertools import product, combinations
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

TARGET_TOP_B = (int(200), int(200))
TARGET_SIDE_B = (int(200), int(200))

ESTIMATE_TOP = (int(200), int(200))
ESTIMATE_SIDE = (int(200), int(200))

SEND_MESSAGES = False

MAG_THRESHOLD = 10
FRAME_THRESHOLD = 5
#
# STATE_NO_TARGET_POINTS = 0
# STATE_ONE_TARGET_POINT_SET = 1
# STATE_SEND_DATA = 2
# STATE_NO_DATA = 3

# STATE = STATE_NO_TARGET_POINTS

def main():
    global SEND_MESSAGES
    global STATE
    global load_video_path

    global use_connection
    global use_recorded_video
    global load_video_path
    global save_video
    global use_target_segmentation
    global use_arduino
    global no_registration

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

    # camera_top_expected_heading = 45
    # camera_side_expected_heading = 45

    output_path = output_dir + output_prefix + '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    print(output_path)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if use_connection:
        print('Connecting to ' + ip_address + ' port ' + str(port) + '...')
        s.connect((ip_address, port))

    arduino = None
    if use_arduino:
        arduino = serial.Serial('/dev/ttyACM2', 19200, timeout=.5)

    bashCommand = 'mkdir -p ' + output_path
    process4 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    cv2.waitKey(100)

    if not use_recorded_video:
        # For both cameras, turn off autofocus and set the same absolute focal depth the one used during calibration.
        # command = 'v4l2-ctl -d /dev/video1 -c focus_auto=0'
        # process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        # cv2.waitKey(100)
        # command = 'v4l2-ctl -d /dev/video1 -c focus_absolute=' + str(camera_top_focus_absolute)
        # command = 'v4l2-ctl -d /dev/video1 -c focus_auto=0 focus_absolute=' + str(camera_top_focus_absolute)\
        #           + ' contrast='+ str(camera_top_contrast) + ' brightness='+ str(camera_top_brightness)\
        #           + ' -d /dev/video2 -c focus_auto=0 focus_absolute=' + str(camera_side_focus_absolute)\
        #           + ' contrast=' + str(camera_side_contrast) + ' brightness='+ str(camera_side_brightness)\
        #           + ' v4l2-ctl -d /dev/video3 -c focus_auto=0 focus_absolute=60'

        command = 'v4l2-ctl -d /dev/video' + str(index_camera_top) + ' -c focus_auto=0'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        # command = 'v4l2-ctl -d /dev/video' + str(index_camera_top) + ' -c focus_absolute=' + str(camera_top_focus_absolute)
        # process1 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        # cv2.waitKey(100)
        command = 'v4l2-ctl -d /dev/video' + str(index_camera_side) + ' -c focus_auto=0'
        process2 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        # command = 'v4l2-ctl -d /dev/video' + str(index_camera_side) + ' -c focus_absolute=' + str(camera_side_focus_absolute)
        # process3 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        # cv2.waitKey(100)
        command1 = 'v4l2-ctl -d /dev/video1 -c focus_absolute=' + str(camera_top_focus_absolute)
        # print(command1)
        process1 = subprocess.Popen(command1.split(), stdout=subprocess.PIPE)
        # cv2.waitKey(100)
        command2 = 'v4l2-ctl -d /dev/video2 -c focus_absolute=' + str(camera_side_focus_absolute)
        # print(command2)
        process1 = subprocess.Popen(command2.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        command3 = 'v4l2-ctl -d /dev/video' + str(index_camera_top) + ' -c contrast='+ str(camera_top_contrast) + ' brightness='+ str(camera_top_brightness)
        process2 = subprocess.Popen(command3.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        command4 = 'v4l2-ctl -d /dev/video' + str(index_camera_side) + ' -c contrast=' + str(camera_side_contrast) + ' brightness='+ str(camera_side_brightness)
        process3 = subprocess.Popen(command4.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)

        cap_top = cv2.VideoCapture(index_camera_top)  # Top camera
        cap_side = cv2.VideoCapture(index_camera_side)  # Side camera
    else:
        # If live video isn't available, use recorded insertion video
        cap_top = cv2.VideoCapture(load_video_path[0] + '/output_top.avi')
        cap_side = cv2.VideoCapture(load_video_path[0] + '/output_side.avi')

    cap_aux = cv2.VideoCapture(index_camera_aux)

    # Load stereo calibration data
    # calibration = np.load('calibration_close.npz')
    # calibration = np.load('calibration.npz')
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

    mat_right_obj = Struct(**cal_right.camera_matrix)
    mat_right = np.reshape(np.array(mat_right_obj.data),(mat_right_obj.rows,mat_right_obj.cols))

    # p_left = Struct(**Struct(**cal_left).projection_matrix)
    # print(np.reshape(np.array(p_left.data),(p_left.rows,p_left.cols)))
    #
    # p_right = Struct(**Struct(**cal_right).projection_matrix)
    # print(np.reshape(np.array(p_right.data),(p_right.rows,p_right.cols)))

    # trans_right = np.array([[-0.0016343138898400025], [-0.13299820438398743], [0.1312384027069722]])
    # trans_right = np.array([[0.0003711532223565725], [-0.1319298883713302], [0.14078849901180754]])
    # trans_right = np.array([[0.0], [-0.130], [0.125]])

    # rot_right = np.array([0.9915492807737206, 0.03743949685116827, -0.12421073976371574, 0.121307736
    # 50921836, 0.07179373377171916, 0.9900151982945141, 0.04598322368134065, -0.9967165815148494, 0.06664532446634884]).reshape((3,3))
    # rot_right = np.array([1.0, 0.0, 0.0,
    #                       0.0, 0.0, 1.0,
    #                       0.0, -1.0, 0.0]).reshape((3,3))


    # p1 = np.concatenate((np.dot(mat_right, np.eye(3)), np.dot(mat_right, np.zeros((3,1)))), axis=1)
    # p2 = np.concatenate((np.dot(mat_right, rot_right), np.dot(mat_right, trans_right)), axis=1)

    # translation_top_to_side = np.array([0.001977984910074812, 0.11807894739709818, 0.1304094431444622]).reshape((3,1))
    # rotation_top_to_side = np.array([0.9991609150516495, -0.018406245002353022, -0.03658792120446591,
    #                                  -0.03783477948348375, -0.07270386689546655, -0.996635679272964,
    #                                  0.015684237137553392, 0.9971837132060268, -0.07333925839585068]).reshape((3,3))

    translation_top_to_side = np.array([9.336674963296142e-05, 0.1268878884308696, 0.12432346740907979]).reshape((3,1))

    rotation_top_to_side = np.array( [0.997701828954437, -0.03188640457921707, -0.0597855977973143,
                                      -0.058907963785628806, 0.027788165175257316, -0.9978765803839791,
                                      0.03348002842894244, 0.9991051371498414, 0.025845940052435786]).reshape((3,3))

    transform_top_to_side = make_homogeneous_tform(rotation=rotation_top_to_side, translation=translation_top_to_side)
    transform_side_to_top = np.linalg.inv(transform_top_to_side)
    print("top to side")
    print(transform_top_to_side)

    print("side to top")
    print(transform_side_to_top)

    p1 = np.dot(mat_left, np.concatenate((np.eye(3), np.zeros((3,1))), axis=1))
    p2 = np.dot(mat_right, transform_side_to_top[0:3,:])

    # p1 = calibration['P1']
    # p2 = calibration['P2']
    # print(p1)
    # print(p2)

    # F = calibration['F']

    # CameraMatrix1 = calibration['CameraMatrix1']
    # DistCoeffs1 = calibration['DistCoeffs1']
    #
    # CameraMatrix2 = calibration['CameraMatrix2']
    # DistCoeffs2 = calibration['DistCoeffs2']

    top_frames = deque(maxlen=3)
    side_frames = deque(maxlen=3)

    transforms_tip = deque(maxlen=2)

    ret, camera_top_last_frame = cap_top.read()
    ret, camera_side_last_frame = cap_side.read()

    top_frames.append(camera_top_last_frame)
    side_frames.append(camera_side_last_frame)

    # camera_top_last_frame = cv2.undistort(camera_top_last_frame, CameraMatrix1, DistCoeffs1)
    # camera_side_last_frame = cv2.undistort(camera_side_last_frame, CameraMatrix2, DistCoeffs2)

    camera_top_height, camera_top_width, channels = camera_top_last_frame.shape
    camera_side_height, camera_side_width, channels = camera_side_last_frame.shape

    camera_top_roi_size = (200, 200)
    camera_side_roi_size = (200, 200)
    # camera_top_roi_size = (camera_top_width, camera_top_height)
    # camera_side_roi_size = camera_top_roi_size

    camera_top_roi_center = (int(camera_top_width * 0.8), camera_top_height / 2)
    camera_side_roi_center = (int(camera_side_width * 0.8), camera_side_height / 2)

    # camera_top_roi_center = (camera_top_width/2, camera_top_height/2)
    # camera_side_roi_center = camera_top_roi_center

    delta_last = None
    position_tip_last = None

    trajectory = []

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
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=20.0,
        frameSize=(camera_top_width, camera_top_height),
        isColor=True)

    out_side = cv2.VideoWriter(
        filename=output_path + '/output_side.avi',
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=20.0,
        frameSize=(camera_side_width, camera_side_height),
        isColor=True)

    out_flow = cv2.VideoWriter(
        filename=output_path + '/output_flow.avi',
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=10.0,
        frameSize=(camera_top_roi_size[0] * 2, camera_top_roi_size[1] * 2),
        isColor=True)

    cv2.namedWindow("Camera Top")
    cv2.namedWindow("Camera Side")

    cv2.setMouseCallback("Camera Top", get_coords_top)
    cv2.setMouseCallback("Camera Side", get_coords_side)

    frames_since_update = 0

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

    tracker_top = tracking.TipTracker(camera_top_farneback_parameters, camera_top_width, camera_top_height,
                             hue_motion, hue_motion_range, int(root.find("threshold_mag").text),
                             camera_top_roi_center, camera_top_roi_size,
                             int(root.find("kernel_top").text), "camera_top", verbose=False)
    tracker_side = tracking.TipTracker(camera_side_farneback_parameters, camera_side_width, camera_side_height,
                                       hue_motion, hue_motion_range, int(root.find("threshold_mag").text),
                              camera_side_roi_center, camera_side_roi_size,
                              int(root.find("kernel_side").text), "camera_side", verbose=False)

    phantom_dims = np.array([0.25, 0.0579, 0.0579]) # length is actually 0.12675 meters
    phantom_transform = np.eye(4)
    phantom_transform[2,3]=0.12
    camera_a_origin = np.array([0,0,0])
    camera_b_origin = trans_right
    compensator_tip = refraction.RefractionModeler(camera_a_origin, np.ravel(camera_b_origin), phantom_dims, phantom_transform, 1.2, 1.0)
    compensator_target = refraction.RefractionModeler(camera_a_origin, np.ravel(camera_b_origin), phantom_dims, phantom_transform, 1.2, 1.0)



    print("Target seg hue range: " + str(target_hue_min) + " to " + str(target_hue_max))
    target_top = tracking.TargetTracker(target_hue_min, target_hue_max, target_sat_min, target_sat_max,
                                        target_val_min, target_val_max, None, TARGET_TOP_A)
    target_side = tracking.TargetTracker(target_hue_min, target_hue_max, target_sat_min, target_sat_max,
                                         target_val_min, target_val_max, None, TARGET_SIDE_A)

    triangulator_tip = tracking.Triangulator(p1, p2)
    triangulator_target = tracking.Triangulator(p1, p2)

    # plotter = plot3dClass(5,5)

    time_last = time.clock()


    while cap_top.isOpened():
        try:
            times = []
            ret, camera_top_current_frame = cap_top.read()
            ret, camera_side_current_frame = cap_side.read()
            # ret, aux_frame = cap_aux.read()
            aux_frame = None

            if cv2.waitKey(1) == ord('q') or camera_top_current_frame is None or camera_side_current_frame is None:
                break

            top_frames.append(camera_top_current_frame)
            side_frames.append(camera_side_current_frame)

            tracker_side.update(side_frames)
            tracker_top.update(top_frames)

            time_delta = time.clock() - time_last
            time_last = time.clock()
            times.append(time_delta)
            print("2D Localization: " + str(time_delta))

            if use_target_segmentation:
                target_top.update(camera_top_current_frame)
                target_side.update(camera_side_current_frame)
                cv2.imshow("Target top", target_top.image_masked)
                cv2.imshow("Target side", target_side.image_masked)
            else:
                target_top.target_coords = TARGET_TOP_A
                target_side.target_coords = TARGET_SIDE_A

            camera_top_with_marker = draw_tip_marker(camera_top_current_frame, tracker_top.roi_center,
                                                     tracker_top.roi_size, tracker_top.position_tip)
            camera_top_with_marker = draw_target_markers(camera_top_with_marker, target_top.target_coords, TARGET_TOP_B)

            camera_side_with_marker = draw_tip_marker(camera_side_current_frame, tracker_side.roi_center,
                                                      tracker_side.roi_size, tracker_side.position_tip)
            camera_side_with_marker = draw_target_markers(camera_side_with_marker, target_side.target_coords, TARGET_SIDE_B)

            position_tip = triangulator_tip.get_position_3D(tracker_top.position_tip, tracker_side.position_tip)
            position_target = triangulator_target.get_position_3D(target_top.target_coords, target_side.target_coords)
            position_target_second = triangulator_target.get_position_3D(TARGET_TOP_B, TARGET_SIDE_B)

            success_compensation_tip, position_tip_corrected_list = compensator_tip.solve_real_point_from_refracted(np.ravel(position_tip))
            success_compensation_target, position_target_corrected_list = compensator_target.solve_real_point_from_refracted(np.ravel(position_target))
            # print(success_compensation_target, position_target_corrected)
            position_tip_corrected = position_tip_corrected_list
            # position_target_corrected = np.array([position_target_corrected_list[0], position_target_corrected_list[1], position_target_corrected_list[2]]).reshape((3,1))
            position_target_corrected = position_target

            time_delta = time.clock() - time_last
            time_last = time.clock()
            times.append(time_delta)
            print("Triangulation/Refraction: " + str(time_delta))

            # position_tip_corrected = np.reshape(position_tip_corrected_temp,(3,1))
            # success_compensation, position_target_corrected = np.reshape(compensator.solve_real_point_from_refracted(np.ravel(position_target)),(3,1))
            # if success_compensation:
            #     compensator_tip.make_plot()
            #
            # success_compensation, position_target_corrected = compensator_target.solve_real_point_from_refracted(np.ravel(position_target))
            # if success_compensation:
            #     compensator_target.make_plot()

            # print("Position top raw", position_tip)
            # print("Position tip corrected", position_tip_corrected)

            # SEND_MESSAGES = True

            print(position_target, position_target_corrected)
            transform_camera_to_target_uncorrected = make_homogeneous_tform(translation=position_target)
            transform_camera_to_target = make_homogeneous_tform(translation=position_target_corrected)
            # transform_camera_to_target_second_uncorrected = make_homogeneous_tform(translation=position_target_second)
            # print("Camera to Target Uncorrected")
            # print(transform_camera_to_target_uncorrected)
            print("Camera to Reg Marker")
            print(transform_camera_to_registration_marker)

            print("Camera to First Target Uncorrected")
            print(transform_camera_to_target_uncorrected)
            # print("Camera to Second Target Uncorrected")
            # print(transform_camera_to_target_second_uncorrected)
            # print("Distance Between Targets: " + str(np.linalg.norm(transform_camera_to_target_uncorrected[:3,3] - transform_camera_to_target_second_uncorrected[:3,3])))

            # print("Camera to Target Corrected")
            # print(transform_camera_to_target)
            print("Camera to First Target Corrected")
            print(transform_camera_to_target)

            transform_registration_marker_to_camera = np.linalg.inv(transform_camera_to_registration_marker)
            print("Reg Marker to Camera")
            print(transform_registration_marker_to_camera)

            transform_registration_marker_to_target = np.dot(transform_registration_marker_to_camera, transform_camera_to_target)
            transform_registration_marker_to_target[0:3,0:3] = np.eye(3)

            print("Reg Marker to Target")
            print(transform_registration_marker_to_target)

            if use_connection:
                # print("Sent!")
                s.send(compose_OpenIGTLink_message(transform_registration_marker_to_target))

            if not np.array_equal(position_tip_corrected, position_tip_last) and success_compensation_tip:
                if arduino is not None:
                    arduino.write('1\n')
                delta = position_target_corrected - position_tip_corrected
                rotation_tip = np.array([[0.99, 0, 0.1], [0.01, 0.99, 0], [0, 0.01, 0.99]])
                if len(transforms_tip) is not 0:
                    direction_motion = normalize(
                        position_tip_corrected.reshape((3, 1)) - transforms_tip[-1][0:3, 3].reshape((3, 1)))
                    axis_y = np.array([0, 1, 0]).reshape((1,3))
                    axis_z = normalize(np.cross(direction_motion.reshape((1,3)), axis_y).reshape((1,3)))
                    axis_y = normalize(np.cross(axis_z.reshape((1,3)), direction_motion.reshape((1,3))))
                    rotation_tip = np.concatenate((direction_motion.reshape((3, 1)), axis_y.reshape((3, 1)), axis_z.reshape((3, 1))), axis=1)

                transform_camera_to_tip = make_homogeneous_tform(rotation=rotation_tip, translation=position_tip_corrected)

                transforms_tip.append(transform_camera_to_tip)

                transform_registration_marker_to_tip = np.dot(np.linalg.inv(transform_camera_to_registration_marker), transform_camera_to_tip)


                print("Marker to Tip")
                print(transform_registration_marker_to_tip)

                print("Tip to Target")
                print(np.dot(np.linalg.inv(transform_registration_marker_to_tip),transform_registration_marker_to_target))

                if use_connection:
                    # print("Sent!")
                    s.send(compose_OpenIGTLink_message(transform_registration_marker_to_tip))
                    # s.send(compose_OpenIGTLink_message(transform_registration_marker_to_target))
                # print('Target: ' + str(target3D))
                # print('Delta: ' + str(delta))
                #
                # print('Target tform: ' + str(transform_to_robot_coords(target3D)))
                # print('Delta tform: ' + str(transform_to_robot_coords(delta)))
                #
                # plotter.drawNow(position_tip)

                position_tip_time = np.concatenate(([[time.clock()]], position_tip_corrected.reshape((3,1))))
                # print(position_tip_time)
                trajectory.append(position_tip_time)
                # print("Adding point to path")
                top_path.append(tracker_top.position_tip)
                side_path.append(tracker_side.position_tip)

            camera_top_with_marker = draw_tip_path(camera_top_with_marker, top_path)
            camera_side_with_marker = draw_tip_path(camera_side_with_marker, side_path)

            # print("Pose tip", pose_tip)
            # print("Pose target", pose_target)

            time_delta = time.clock() - time_last
            time_last = time.clock()
            times.append(time_delta)
            print("Comms: " + str(time_delta))

            cv2.imshow('Camera Top', camera_top_current_frame)
            cv2.imshow('Camera Side', camera_side_current_frame)

            # cv2.imshow("Flow Mag Top", tracker_top.flow_mag)
            # cv2.imshow("Flow Mag Side", tracker_side.flow_mag)
            #
            # cv2.imshow("Cam Top Thresh", tracker_top.image_current_gray_thresh)
            # cv2.imshow("Cam Side Thresh", tracker_side.image_current_gray_thresh)

            font = cv2.FONT_HERSHEY_DUPLEX
            text_color = (0, 255, 0)
            data_frame = np.zeros_like(camera_top_with_marker)

            cv2.putText(data_frame, 'Delta: ' + make_data_string(delta),
                        (10, 50), font, 1, text_color)

            cv2.putText(data_frame, 'Target: ' + make_data_string(position_target_corrected),
                        (10, 100), font, 1, text_color)

            cv2.putText(data_frame, 'Tip: ' + make_data_string(trajectory[-1][1:,:].reshape((3,1))),
                        (10, 150), font, 1, text_color)

            cv2.putText(data_frame, 'Top  2D: ' + str(tracker_top.position_tip[0]) + ' ' + str(tracker_top.position_tip[1]),
                        (10, 200), font, 1, text_color)

            cv2.putText(data_frame,
                        'Side 2D: ' + str(tracker_side.position_tip[0]) + ' ' + str(tracker_side.position_tip[1]),
                        (10, 250), font, 1, text_color)

            if aux_frame is not None:
                combined2 = np.concatenate((data_frame, aux_frame), axis=0)
            else:
                combined2 = np.concatenate((data_frame, np.zeros_like(data_frame)), axis=0)

            out_top.write(camera_top_current_frame)
            out_side.write(camera_side_current_frame)

            combined1 = np.concatenate((camera_top_with_marker, camera_side_with_marker), axis=0)
            combined = np.array(np.concatenate((combined1, combined2), axis=1), dtype=np.uint8)

            combined_flow = np.array(np.concatenate((tracker_top.flow_diagnostic, tracker_side.flow_diagnostic), axis=1), dtype=np.uint8)
            cv2.imshow('Combined', combined)
            cv2.imshow('Combined Flow', combined_flow)
            out_combined.write(combined)
            out_flow.write(combined_flow)

            delta_last = delta
            position_tip_last = position_tip_corrected

            time_delta = time.clock() - time_last
            time_last = time.clock()
            times.append(time_delta)
            print("Diagnostics: " + str(time_delta))
            print("Total: " + str(sum(times)) + "\n")
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

    trajectoryArray = np.array(trajectory)
    # print(trajectoryArray)
    np.savetxt(output_path + "/trajectory.csv", trajectoryArray, delimiter=",")
    np.savez_compressed(output_path + "/trajectory.npz", trajectory=trajectoryArray,
                        top_path=np.array(top_path), side_path=np.array(side_path))



class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class plot3dClass(object):
    def __init__(self, systemSideLength, lowerCutoffLength ):
        self.systemSideLength = systemSideLength
        self.lowerCutoffLength = lowerCutoffLength

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # rng = np.arange(0, self.systemSideLength, self.lowerCutoffLength)
        # self.X, self.Y = np.meshgrid(rng, rng)

        self.ax.w_zaxis.set_major_locator(LinearLocator(10))
        self.ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

        # heightR = np.zeros(self.X.shape)
        # self.surf = self.ax.plot_surface(
        #     self.X, self.Y, heightR, rstride=1, cstride=1,
        #     cmap=cm.jet, linewidth=0, antialiased=False)
        # plt.draw() maybe you want to see this frame?

    def drawNow(self, point):

        # self.surf.remove()
        # self.surf = self.ax.plot_surface(
        #     self.X, self.Y, heightR, rstride=1, cstride=1,
        #     cmap=cm.jet, linewidth=0, antialiased=False)

        r = [-0.05, 0.05]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                self.ax.plot3D(*zip(s, e), color="b")
        # print(point)
        self.ax.scatter(point[0], point[1], point[2], color="r")

        plt.draw()  # redraw the canvas

        self.fig.canvas.flush_events()
        # time.sleep(1)

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

def get_coords_top(event, x, y, flags, param):
    # global STATE
    global TARGET_TOP_A
    global TARGET_TOP_B
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Click in top image")
        TARGET_TOP_A = x, y
        # if STATE == STATE_NO_TARGET_POINTS:
        #     STATE = change_state(STATE, STATE_ONE_TARGET_POINT_SET)
        # elif STATE == STATE_ONE_TARGET_POINT_SET:
        #     STATE == change_state(STATE, STATE_SEND_DATA)
        # elif STATE == STATE_SEND_DATA:
        #     STATE == change_state(STATE, STATE_ONE_TARGET_POINT_SET)

    elif event == cv2.EVENT_MBUTTONDOWN:
        print("Right click in top image")
        TARGET_TOP_B = x,y

def get_coords_side(event, x, y, flags, param):
    # global STATE
    global TARGET_SIDE_A
    global TARGET_SIDE_B
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Click in side image")
        TARGET_SIDE_A = x, y
        # if STATE == STATE_NO_TARGET_POINTS:
        #     STATE = change_state(STATE, STATE_ONE_TARGET_POINT_SET)
        # elif STATE == STATE_ONE_TARGET_POINT_SET:
        #     STATE == change_state(STATE, STATE_SEND_DATA)
        # elif STATE == STATE_SEND_DATA:
        #     STATE == change_state(STATE, STATE_ONE_TARGET_POINT_SET)
    elif event == cv2.EVENT_MBUTTONDOWN:
        print("Right click in side image")
        TARGET_SIDE_B = x,y

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
    # TODO: Make this smarter for different types of inputs, or find a built-in in numpy or opencv
    homogeneous = np.eye(4)
    homogeneous[0:3, 0:3] = rotation
    homogeneous[0:3, 3] = translation.reshape((3,1))[:,0]
    return homogeneous

def compose_OpenIGTLink_message(input_tform):
    body = struct.pack('!12f',
                       float(input_tform[0,0]), float(input_tform[1,0]), float(input_tform[2,0]),
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

# def change_state(current_state, new_state):
#     if current_state == STATE_NO_TARGET_POINTS:
#         if new_state == STATE_ONE_TARGET_POINT_SET:
#             return new_state
#
#     elif current_state == STATE_ONE_TARGET_POINT_SET:
#         if new_state == STATE_SEND_DATA or new_state == STATE_NO_TARGET_POINTS:
#             return new_state
#
#     elif current_state == STATE_SEND_DATA:
#         if new_state == STATE_NO_DATA or new_state == STATE_ONE_TARGET_POINT_SET:
#             return new_state
#
#     elif current_state == STATE_NO_DATA:
#         if new_state == STATE_SEND_DATA or new_state == STATE_ONE_TARGET_POINT_SET:
#             return new_state
#
#     else:
#         return current_state
#
# def print_state(current_state):
#     if current_state == STATE_NO_TARGET_POINTS:
#         print('STATE_NO_TARGET_POINTS')
#     elif current_state == STATE_ONE_TARGET_POINT_SET:
#         print('STATE_ONE_TARGET_POINT_SET')
#     elif current_state == STATE_SEND_DATA:
#         print('STATE_SEND_DATA')
#     elif current_state == STATE_NO_DATA:
#         print('STATE_NO_DATA')

def make_data_string(data):
    return '%0.3g, %0.3g, %0.3g' % (data.ravel()[0], data.ravel()[1], data.ravel()[2])

if __name__ == '__main__':
    main()
