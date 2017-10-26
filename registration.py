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

# Parse commang line arguments. These are primarily flags for things likely to change between runs.
parser = argparse.ArgumentParser(description='Register cameras and phantom to global coordinate frame.')
parser.add_argument('--use_connection', action='store_true',
                    help='Attempt to connect to the robot control computer.')
parser.add_argument('--use_recorded_video', action='store_true',
                    help='Load and process video from file, instead of trying to get live video from webcams.')
parser.add_argument('--load_video_path', type=str, nargs=1, default='./data/test',
                    help='Path for video to load if --use_recorded_video is specified.')
parser.add_argument('--square_size', type=float, nargs=1, default=0.0060175,
                    help='Calibration checkerboard square edge length')
args = parser.parse_args()
globals().update(vars(args))

tree = ET.parse('config.xml')
root = tree.getroot()
ip_address = str(root.find("ip").text)
port = int(root.find("port").text)

def main():
    global STATE
    # TODO: Load a primitive rectangular prism representing the phantom

    if not use_recorded_video:
        # For both cameras, turn off autofocus and set the same absolute focal depth the one used during calibration.
        command = 'v4l2-ctl -d /dev/video1 -c focus_auto=0'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        command = 'v4l2-ctl -d /dev/video1 -c focus_absolute=20'
        process1 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        command = 'v4l2-ctl -d /dev/video2 -c focus_auto=0'
        process2 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        command = 'v4l2-ctl -d /dev/video2 -c focus_absolute=60'
        process3 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)

        cap_top = cv2.VideoCapture(1)  # Top camera
        cap_side = cv2.VideoCapture(2)  # Side camera
    else:
        # If live video isn't available, use recorded insertion video
        cap_top = cv2.VideoCapture(str(load_video_path + '/output_top.avi'))
        cap_side = cv2.VideoCapture(str(load_video_path + '/output_side.avi'))

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if use_connection:
        print('Connecting to ' + ip_address + ' port ' + str(port) + '...')
        s.connect((ip_address, port))

    cal_left = Struct(**yaml.load(file('left.yaml','r')))
    cal_right = Struct(**yaml.load(file('right.yaml', 'r')))

    mat_left = yaml_to_mat(cal_left.camera_matrix)
    mat_right= yaml_to_mat(cal_right.camera_matrix)
    dist_left = yaml_to_mat(cal_left.distortion_coefficients)
    dist_right = yaml_to_mat(cal_right.distortion_coefficients)

    trans_right = np.array([[-0.0016343138898400025], [-0.13299820438398743], [0.1312384027069722]])
    rot_right = np.array([0.9915492807737206, 0.03743949685116827, -0.12421073976371574, 0.12130773650921836, 0.07179373377171916, 0.9900151982945141, 0.04598322368134065, -0.9967165815148494, 0.06664532446634884]).reshape((3,3))

    p1 = np.concatenate((np.dot(mat_left, np.eye(3)), np.dot(mat_left, np.zeros((3,1)))), axis=1)
    p2 = np.concatenate((np.dot(mat_right, rot_right), np.dot(mat_right, trans_right)), axis=1)

    cv2.namedWindow("Camera Top")
    cv2.namedWindow("Camera Side")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)


    # x = [0, 0.1]
    # y = [0, 0.05]
    # z = [0, 0.05]
    # # r = [-0.05, 0.05]
    # for s, e in combinations(np.array(list(product(x, y, z))), 2):
    #     # if np.sum(np.abs(s - e)) == r[1] - r[0]:
    #         # self.ax.plot3D(*zip(s, e), color="b")
    #     print(s,e)

    # square edge length (m) = 0.0060175

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    print(mat_left)
    print(dist_left)

    transform_homogeneous = np.zeros((4,4))

    # dpi = 200.0
    square_width = 0.007
    marker_width = square_width * 0.5
    squares_wide = 7
    squares_high = 9
    # in_per_m = 1 / 2.54 * 100
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    board = cv2.aruco.CharucoBoard_create(squares_wide, squares_high, square_width, marker_width, dictionary)

    while cap_top.isOpened():
        ret, frame_top = cap_top.read()
        ret, frame_side = cap_side.read()

        if cv2.waitKey(10) == ord('q') or frame_top is None or frame_side is None:
            break

        frame_top_markers = frame_top
        frame_side_markers = frame_side

        # TODO: Pick three known points on the phantom in the side camera image
        # TODO: Draw a wireframe box representing the phantom on the side camera image to show phantom registration
        # Register phantom using solvePnPRansac, with the object points being the coordinates of the mesh vertices
        # and the image points being the corresponding pixel coordinates in the side camera image.
        # Need to get a library that does intersections between primitives and rays (trimesh?)
        # Need a good way to specify phantom dimensions (some kind of config file?) and import
        # TODO: Find the pose of a checkerboard image
        # TODO: Solve for the transform between the checkerboard and the origin of the stereo camera pair


        gray = cv2.cvtColor(frame_side, cv2.COLOR_BGR2GRAY)


        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(image=frame_side, dictionary=dictionary)
        # print(markerIds)
        if markerIds is not None:
            count, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(markerCorners=markerCorners, markerIds=markerIds, image=frame_side, board=board, cameraMatrix=mat_left, distCoeffs=dist_left)
            # print(charucoCorners)
            # print(charucoIds)
            # print("stuff!", ret)
            cv2.aruco.drawDetectedCornersCharuco(image=frame_side, charucoCorners=charucoCorners, charucoIds=charucoIds)

            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners=charucoCorners, charucoIds=charucoIds, board=board, cameraMatrix=mat_left, distCoeffs=dist_left)
            # project 3D points to image plane
            # print(rvec, tvec)
            # print(rvec, '\n')
            if ret:
                print(rvec)
                rmat, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
                transform_homogeneous = np.concatenate(
                    (np.concatenate((rmat, tvec), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
                print(transform_homogeneous)
            # imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mat_left, dist_left)
            # frame_side_markers = draw(frame_side, charucoCorners, imgpts)
            # cv2.imshow('frame_side_markers', frame_side_markers)


        # ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
        #
        # if ret == True:
        #     print("Found corners")
        #     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #
        #     # Find the rotation and translation vectors.
        #     print("objp", objp)
        #     print("corners2",corners2)
        #     ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mat_left, dist_left)
        #
        #     rmat, _ = cv2.Rodrigues(rvecs)
        #
        #     transform_homogeneous = np.concatenate((np.concatenate((rmat, tvecs*square_size), axis=1), np.array([[0,0,0,1]])), axis=0)
        #     print(transform_homogeneous)
        #
        #     # project 3D points to image plane
        #     imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mat_left, dist_left)
        #     frame_side_markers = draw(frame_side, corners2, imgpts)
        #     cv2.imshow('frame_side_markers', frame_side_markers)
        #
        #
        # markers_phantom = find_phantom_markers(frame_side)
        # points_phantom = np.array([[0,0,0],[0,5.16,0],[3.9,0,0], [7.75,0,0]],dtype=np.float32)
        #
        # # print("object points",points_phantom)
        # # print("markers",markers_phantom)
        # ret, rvecs_phantom, tvecs_phantom, inliers_phantom = cv2.solvePnPRansac(points_phantom, markers_phantom, mat_left, dist_left)
        # print("rvecs_phantom", rvecs_phantom)
        # print("tvecs_phantom", tvecs_phantom*0.01)
        # # project 3D points to image plane
        # imgpts_phantom, jac = cv2.projectPoints(axis, rvecs_phantom, tvecs_phantom, mat_left, dist_left)
        # frame_side_markers_phantom = draw(frame_side, markers_phantom, imgpts_phantom)
        # cv2.imshow('frame_side_markers_phantom', frame_side_markers_phantom)


        cv2.imshow('Camera Top', frame_top_markers)
        cv2.imshow('Camera Side', frame_side_markers)

    if use_connection:
        s.send(make_OIGTL_homogeneous_tform(transform_homogeneous))

    cap_top.release()
    cap_side.release()

    cv2.destroyAllWindows()

    if s is not None:
        s.close()

class Triangulator:
    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2

    def _to_float(self, coords):
        return (float(coords[0]), float(coords[1]))

    def get_position_3D(self, coords_top, coords_side):
        pose_3D_homogeneous = cv2.triangulatePoints(self.P1, self.P2,
                                                    np.array(self._to_float(coords_top)).reshape(2, -1),
                                                    np.array(self._to_float(coords_side)).reshape(2, -1))
        return (pose_3D_homogeneous / pose_3D_homogeneous[3])[0:3]

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def find_phantom_markers(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    segmented = cv2.inRange(image_hsv, np.array([0,0,0]), np.array([180, 255, 80]))
    cv2.imshow("Seg", segmented)
    img, contours, hierarchy = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = image
    markers = []
    if len(contours) > 0:
        areas = []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areas.append(area)
        contours_sorted = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)

        cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 3)
        cv2.imshow("Contours", image_contours)
        # print(contours_sorted)

        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0.0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                markers.append([[cx, cy]])
    return np.array(markers, dtype=np.float32)


def yaml_to_mat(input):
    obj = Struct(**input)
    return np.reshape(np.array(obj.data),(obj.rows,obj.cols))

def draw_target_marker(image, target_coords):
    output = image.copy()
    cv2.circle(output, target_coords, 10, (0, 255, 0))
    return output

def transform_to_robot_coords(input):
    return np.array([-input[2], input[1], -input[0]])

def make_OIGTL_homogeneous_tform(input_tform):
    body = struct.pack('!12f',
                       float(input_tform((0,0))), float(input_tform((1,0))), float(input_tform((2,0))),
                       float(input_tform((0, 1))), float(input_tform((1, 1))), float(input_tform((2, 1))),
                       float(input_tform((0, 2))), float(input_tform((1, 2))), float(input_tform((2, 2))),
                       float(input_tform((0, 3))), float(input_tform((1, 3))), float(input_tform((2, 3))))
    bodysize = 48
    return struct.pack('!H12s20sIIQQ', 1, str('TRANSFORM'), str('SIMULATOR'), int(time.time()), 0, bodysize, 0) + body


if __name__ == '__main__':
    main()
