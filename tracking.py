import cv2
import numpy as np
import itertools
from collections import deque

class TipTracker:
    def __init__(self, params, image_width, image_height, heading_expected,
                 heading_range, threshold_mag_lower, threshold_mag_upper, roi_center_initial,
                 roi_size, kernel_size, name="camera", verbose=False):

        self.flow_params = params
        self.heading = heading_expected
        self.heading_range = heading_range
        self.roi_center = roi_center_initial
        self.roi_size = roi_size
        self.image_width = image_width
        self.image_height = image_height
        self.position_tip = roi_center_initial
        self.flow_previous = None
        self.name = name
        self.kernel_size = kernel_size
        self.verbose = verbose
        self.threshold_mag_lower = threshold_mag_lower
        self.threshold_mag_upper = threshold_mag_upper

        self.heading_insert_bound_lower = (self.heading - (self.heading_range / 2))%180
        self.heading_insert_bound_upper = (self.heading + (self.heading_range / 2))%180
        self.heading_retract_bound_lower = (self.heading + 90 - self.heading_range / 2)%180
        self.heading_retract_bound_upper = (self.heading + 90 + self.heading_range / 2)%180
        if verbose:
            print("Insert bounds: " + str(self.heading_insert_bound_lower) + " to " + str(self.heading_insert_bound_upper))
            print("Retract bounds: " + str(self.heading_retract_bound_lower) + " to " + str(self.heading_retract_bound_upper))
            self._show_hue_range(self.heading_insert_bound_lower, self.heading_insert_bound_upper,"insert")
            self._show_hue_range(self.heading_retract_bound_lower, self.heading_retract_bound_upper, "retract")

    def _show_hue_range(self, bound_lower, bound_upper, tag):
        color_range = np.array(np.zeros((500,50,3)),dtype=np.uint8)
        step_count = bound_upper - bound_lower
        step_size = 500/(step_count)
        for value in range(0,step_count):
            color_range[value*step_size:(value+1)*step_size,:,:]=(bound_lower+value, 200, 200)
        color_range_bgr = cv2.cvtColor(color_range, cv2.COLOR_HSV2BGR)
        cv2.imshow(self.name+"_range_"+tag, color_range_bgr)

    def _get_section(self, image):
        return image[self.roi_center[1] - self.roi_size[1] / 2:self.roi_center[1] + self.roi_size[1] / 2,
               self.roi_center[0] - self.roi_size[0] / 2:self.roi_center[0] + self.roi_size[0] / 2]

    def _get_dense_flow(self, image_past, image_current):
        image_past_gray = cv2.cvtColor(image_past, cv2.COLOR_BGR2GRAY)
        image_current_gray = cv2.cvtColor(image_current, cv2.COLOR_BGR2GRAY)

        self.image_current_gray_thresh = cv2.inRange(image_current_gray, 0, 100)

        if self.flow_previous is None:
            flow = cv2.calcOpticalFlowFarneback(image_past_gray,
                                                image_current_gray,
                                                None,
                                                self.flow_params[0], self.flow_params[1], self.flow_params[2],
                                                self.flow_params[3], self.flow_params[4], self.flow_params[5],
                                                self.flow_params[6])
        else:
            flow = cv2.calcOpticalFlowFarneback(image_past_gray,
                                                image_current_gray,
                                                self.flow_previous,
                                                self.flow_params[0], self.flow_params[1], self.flow_params[2],
                                                self.flow_params[3], self.flow_params[4], self.flow_params[5],
                                                self.flow_params[6])# + cv2.OPTFLOW_USE_INITIAL_FLOW)
        self.flow_previous = flow
        flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return flow_magnitude, flow_angle


    def _dense_flow_to_image(self, flow_mag, flow_angle, shape):
        hsv = np.zeros(shape, dtype=np.float32)
        hsv[..., 1] = 255

        hsv[..., 0] = ((flow_angle+(np.pi/2))%(2*np.pi) * (180 / np.pi)) * 0.5
        hsv[..., 2] = flow_mag

        hsv_rescaled = hsv.copy()
        hsv_rescaled[..., 2] = np.clip(hsv_rescaled[..., 2] * (120 / self.threshold_mag_lower), 0, 255)
        bgr = cv2.cvtColor(np.array(hsv_rescaled, dtype=np.uint8), cv2.COLOR_HSV2BGR)
        # print(np.max(flow_magnitude), np.std(flow_magnitude), np.mean(flow_magnitude))
        return hsv, bgr

    def _filter_by_heading(self, flow_hsv):
        flow_hsv_insert_bound_lower = np.array([self.heading_insert_bound_lower, 50, self.threshold_mag_lower])
        flow_hsv_insert_bound_upper = np.array([self.heading_insert_bound_upper, 255, self.threshold_mag_upper])

        mask = cv2.inRange(flow_hsv, flow_hsv_insert_bound_lower, flow_hsv_insert_bound_upper)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=1)
        dilate = cv2.dilate(erosion, kernel, iterations=1)

        ret, thresh = cv2.threshold(dilate, 127, 255, 0)
        return thresh

    def _get_tip_coords(self, image_thresholded):
        position_tip = None

        img, contours, hierarchy = cv2.findContours(image_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            areas = []
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                areas.append(area)
            contours_sorted = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)
            contour_largest = contours_sorted[0][1]

            M = cv2.moments(contour_largest)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            tip_x = self.roi_center[0] - self.roi_size[0] / 2 + cx
            tip_y = self.roi_center[1] - self.roi_size[1] / 2 + cy

            position_tip = (tip_x, tip_y)
        return position_tip

    def _get_new_valid_roi(self, position_tip):
        return (min(max(self.roi_size[0] / 2, position_tip[0]), self.image_width - self.roi_size[0] / 2),
                min(max(self.roi_size[1] / 2, position_tip[1]), self.image_height - self.roi_size[1] / 2))

    def update(self, frames, use_manual_roi=False, manual_roi=(0,0)):
        if use_manual_roi:
            self.roi_center = self._get_new_valid_roi(manual_roi)
        num_frames = len(frames)

        frame_current = frames[-1]
        section_current = self._get_section(frame_current)

        frames_past = deque(itertools.islice(frames, 0, num_frames-1))
        frame_past = frames_past[-1]

        flow_mags = []
        flow_angles = []
        for frame_past in frames_past:
            section_past = self._get_section(frame_past)
            flow_mag, flow_angle = self._get_dense_flow(section_past, section_current)
            flow_mags.append(flow_mag)
            flow_angles.append(flow_angle)

        flow_mag_mean = np.mean(np.array(flow_mags), axis=0)
        flow_angle_mean = np.mean(np.array(flow_angles), axis=0)

        # section_past = self._get_section(frame_past)
        # flow_mag, flow_angle = self._get_dense_flow(section_past, section_current)

        self.flow_hsv, self.flow_bgr = self._dense_flow_to_image(flow_mag_mean, flow_angle_mean, section_current.shape)

        flow_thresholded = self._filter_by_heading(self.flow_hsv)

        self.flow_diagnostic = np.zeros((2 * self.roi_size[1], self.roi_size[0], 3), np.uint8)
        self.flow_diagnostic[:self.roi_size[1], :, :] = self.flow_bgr
        self.flow_diagnostic[self.roi_size[1]:, :, :] = cv2.cvtColor(flow_thresholded, cv2.COLOR_GRAY2BGR)

        position_tip_new = self._get_tip_coords(flow_thresholded)
        if position_tip_new is not None:
            self.position_tip = position_tip_new
            self.roi_center = self._get_new_valid_roi((int(self.position_tip[0]-float(self.roi_size[0]*0.33)), self.position_tip[1]))


class Triangulator:
    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2
        # print("P1")
        # print(self.P1)
        # print("P2")
        # print(self.P2)

    def _to_float(self, coords):
        return (float(coords[0]), float(coords[1]))

    def get_position_3D(self, coords_top, coords_side):
        pose_3D_homogeneous = cv2.triangulatePoints(self.P1, self.P2,
                                                    np.array(self._to_float(coords_side)).reshape(2, -1),
                                                    np.array(self._to_float(coords_top)).reshape(2, -1))
        return (pose_3D_homogeneous / pose_3D_homogeneous[3])[0:3]

class TargetTracker:
    def __init__(self, target_hue_min, target_hue_max, target_sat_min, target_sat_max, target_val_min, target_val_max, dims_window, target_coords_initial):
        self.target_hue_min = target_hue_min
        self.target_hue_max = target_hue_max
        self.target_sat_min = target_sat_min
        self.target_sat_max = target_sat_max
        self.target_val_min = target_val_min
        self.target_val_max = target_val_max
        self.dims_window = dims_window
        self.target_coords = target_coords_initial

    def update(self, image):
        # TODO: localize target as centroid of cluster near specified HSV values

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        bound_lower = np.array([self.target_hue_min/2, self.target_sat_min, self.target_val_min])
        bound_upper = np.array([self.target_hue_max/2, self.target_sat_max, self.target_val_max])

        mask = cv2.inRange(image_hsv, bound_lower, bound_upper)

        kernel = np.ones((7, 7), np.uint8)
        mask_opened = cv2.erode(cv2.dilate(mask, kernel, iterations=1), kernel, iterations=1)
        # mask_opened = mask


        self.image_masked = mask_opened

        img, contours, hierarchy = cv2.findContours(mask_opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            areas = []
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                areas.append(area)
            contours_sorted = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)
            contour_largest = contours_sorted[0][1]

            M = cv2.moments(contour_largest)
            if M['m00'] > 0.0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.target_coords = (cx, cy)

class PhantomTracker:
    def __init__(self, board, dictionary, dims_phantom):
        self.board = board
        self.dictionary = dictionary
        self.dims_phantom = dims_phantom
        self.transform_camera_to_phantom = None
        self.transform_last = None
        self.vertices_phantom = np.array([[-dims_phantom[0], -dims_phantom[1], -dims_phantom[2]],
                                          [-dims_phantom[0], dims_phantom[1], -dims_phantom[2]],
                                          [-dims_phantom[0], -dims_phantom[1], dims_phantom[2]],
                                          [-dims_phantom[0], dims_phantom[1], dims_phantom[2]],
                                          [dims_phantom[0], -dims_phantom[1], -dims_phantom[2]],
                                          [dims_phantom[0], dims_phantom[1], -dims_phantom[2]],
                                          [dims_phantom[0], -dims_phantom[1], dims_phantom[2]],
                                          [dims_phantom[0], dims_phantom[1], dims_phantom[2]]])*0.5
        # self.vertices_phantom = np.array([[              0,               0,               0],
        #                                   [              0, dims_phantom[1],               0],
        #                                   [              0,               0, dims_phantom[2]],
        #                                   [              0, dims_phantom[1], dims_phantom[2]],
        #                                   [dims_phantom[0],               0,               0],
        #                                   [dims_phantom[0], dims_phantom[1],               0],
        #                                   [dims_phantom[0],               0, dims_phantom[2]],
        #                                   [dims_phantom[0], dims_phantom[1], dims_phantom[2]]])

        self.transforms_vertex_phantom = []
        for vertex in self.vertices_phantom:
            self.transforms_vertex_phantom.append(np.concatenate(
                    (np.concatenate((np.eye(3), vertex.reshape((3,1))), axis=1), np.array([[0, 0, 0, 1]])), axis=0))
        # print(self.transforms_vertex_phantom)

    def update(self, image, mat_camera, dist_camera):
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(image=image, dictionary=self.dictionary)
        if markerIds is not None:
            ret, rvec, tvec = cv2.aruco.estimatePoseBoard(corners=markerCorners, ids=markerIds, board=self.board, cameraMatrix=mat_camera, distCoeffs=dist_camera)
            if ret:
                rmat, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
                self.transform_camera_to_phantom = np.concatenate(
                    (np.concatenate((rmat, tvec), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
                # if self.transform_last is None:
                #     self.transform_camera_to_phantom = transform_camera_to_phantom
                #     self.transform_last = transform_camera_to_phantom
                # else:
                #     diff = np.dot(np.linalg.inv(self.transform_last), transform_camera_to_phantom)
                #     mag = np.linalg.norm(diff[0:3,3])
                #     print(mag)
                #     if mag <= 0.01:
                #         self.transform_camera_to_phantom = transform_camera_to_phantom

        print(self.transform_camera_to_phantom)

    def get_phantom_corner_image_points(self, mat_camera, dist_camera, rvec_camera=np.eye(3), tvec_camera=np.zeros((3,1))):
        vertices_transformed = []
        for transform_phantom_to_vertex in self.transforms_vertex_phantom:
            vertices_transformed.append(np.dot(self.transform_camera_to_phantom, transform_phantom_to_vertex))
        # print("Transform camera to phantom")
        # print(self.transform_camera_to_phantom)
        # print("Transform phantom to vertex")
        # print(self.transforms_vertex_phantom)
        # print("Transform camera to vertex")
        # print(np.array(vertices_transformed))
        pts, _ = cv2.projectPoints(objectPoints=np.array(vertices_transformed)[:,0:3,3],
                                   rvec=rvec_camera,
                                   tvec=tvec_camera,
                                   cameraMatrix=mat_camera,
                                   distCoeffs=dist_camera)
        self.image_points = pts.reshape((-1,2)).astype(np.int32)
        # print("Image point")
        print(self.image_points)

    def draw_phantom_axes(self, image, mat_camera, dist_camera):
        return cv2.aruco.drawAxis(image=image,
                                  cameraMatrix=mat_camera,
                                  distCoeffs=dist_camera,
                                  rvec=self.transform_camera_to_phantom[0:3, 0:3],
                                  tvec=self.transform_camera_to_phantom[0:3, 3],
                                  length=0.03)

    def draw_phantom_corners(self, image, mat_camera, dist_camera, rvec_camera=np.eye(3), tvec_camera=np.zeros((3,1))):
        self.get_phantom_corner_image_points(mat_camera, dist_camera, rvec_camera=rvec_camera, tvec_camera=tvec_camera)
        output = image.copy()
        for point in self.image_points:
            cv2.circle(output, (int(point[0]), int(point[1])), 7, (0, 255, 255))
        return output

    def get_phantom_mask(self, shape, mat_camera, dist_camera, rvec_camera=np.eye(3), tvec_camera=np.zeros((3,1))):
        self.get_phantom_corner_image_points(mat_camera, dist_camera, rvec_camera, tvec_camera)
        filled_mask =  cv2.cvtColor(cv2.fillConvexPoly(np.zeros((480,640), dtype=np.uint8), self.image_points, len(self.image_points), 255), cv2.COLOR_GRAY2BGR)
        cv2.imshow("Phantom mask", filled_mask)

