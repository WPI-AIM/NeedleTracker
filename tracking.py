import cv2
import numpy as np

class TipTracker:
    def __init__(self, params, image_width, image_height, heading_expected,
                 heading_range, roi_center_initial, roi_size, kernel_size, name="camera", verbose=False):

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
        # cv2.imshow(self.name+"_gray",image_current_gray)

        self.image_current_gray_thresh = cv2.inRange(image_current_gray, 0, 100)
        # cv2.imshow(self.name+"_thresh", image_past_gray_thresh)

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

        hsv = np.zeros_like(image_current)
        hsv[..., 1] = 255

        hsv[..., 0] = ((flow_angle+90)%360 * (180 / np.pi)) * 0.5
        hsv[..., 2] = flow_magnitude

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # print(np.max(flow_magnitude), np.std(flow_magnitude), np.mean(flow_magnitude))
        return hsv, bgr, flow_magnitude

    def _filter_by_heading(self, flow_hsv):
        max_value = 255
        threshold_magnitude = 5

        flow_hsv_insert_bound_lower = np.array([self.heading_insert_bound_lower, 50, threshold_magnitude])
        flow_hsv_insert_bound_upper = np.array([self.heading_insert_bound_upper, 255, max_value])

        mask_insert = cv2.inRange(flow_hsv, flow_hsv_insert_bound_lower, flow_hsv_insert_bound_upper)

        flow_hsv_retract_bound_lower = np.array([self.heading_retract_bound_lower, 50, threshold_magnitude])
        flow_hsv_retract_bound_upper = np.array([self.heading_retract_bound_upper, 255, max_value])

        mask_retract = cv2.inRange(flow_hsv, flow_hsv_retract_bound_lower, flow_hsv_retract_bound_upper)

        mask = cv2.bitwise_or(mask_insert, mask_retract)

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

    def update(self, frames):
        frame_current = frames[-1]
        frame_past = frames[0]
        section_current = self._get_section(frame_current)
        section_past = self._get_section(frame_past)

        self.flow_hsv, self.flow_bgr, self.flow_mag = self._get_dense_flow(section_past, section_current)

        flow_thresholded = self._filter_by_heading(self.flow_hsv)

        self.flow_diagnostic = np.zeros((2 * self.roi_size[1], self.roi_size[0], 3), np.uint8)
        self.flow_diagnostic[:self.roi_size[1], :, :] = self.flow_bgr
        self.flow_diagnostic[self.roi_size[1]:, :, :] = cv2.cvtColor(flow_thresholded, cv2.COLOR_GRAY2BGR)

        position_tip_new = self._get_tip_coords(flow_thresholded)
        if position_tip_new is not None:
            self.position_tip = position_tip_new
            self.roi_center = self._get_new_valid_roi(self.position_tip)


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

class TargetTracker:
    def __init__(self, target_hsv, target_hsv_range, dims_window, target_coords_initial):
        self.target_hsv = target_hsv
        self.target_hsv_range = target_hsv_range
        self.dims_window = dims_window
        self.target_coords = target_coords_initial

    def update(self, image):
        # TODO: localize target as centroid of cluster near specified HSV values

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        bound_lower = np.array([self.target_hsv/2 - self.target_hsv_range/4, 20, 50])
        bound_upper = np.array([self.target_hsv/2 + self.target_hsv_range/4, 255, 150])

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
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            self.target_coords = (cx, cy)