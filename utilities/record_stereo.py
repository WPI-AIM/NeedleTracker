import numpy as np
import cv2
import time
import subprocess

def main():
	max_count = 18
	delay = 5
	count = 0
	start = time.time()

	SIDE = True
	TOP = True

	boardSize = (9, 7)

	IMG_PATH = '/home/jgschornak/NeedleGuidance/images_converging_cams/'

	bashCommand = 'v4l2-ctl -d /dev/video1 -c focus_auto=0'
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	cv2.waitKey(100)
	bashCommand = 'v4l2-ctl -d /dev/video1 -c focus_absolute=20'
	process1 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	cv2.waitKey(100)
	bashCommand = 'v4l2-ctl -d /dev/video2 -c focus_auto=0'
	process2 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	cv2.waitKey(100)
	bashCommand = 'v4l2-ctl -d /dev/video2 -c focus_absolute=30'
	process3 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	cv2.waitKey(100)

	top = cv2.VideoCapture(1)
	side = cv2.VideoCapture(2)

	ret, frame = top.read()
	height, width, channels = frame.shape

	calibration = np.load('calibration.npz')

	# P1 = calibration['P1']
	# P2 = calibration['P2']

	top_camera_matrix = calibration['CameraMatrix1']
	top_dist_coefs = calibration['DistCoeffs1']

	side_camera_matrix = calibration['CameraMatrix2']
	side_dist_coefs = calibration['DistCoeffs2']  

	tvec_top = None
	tvec_side = None
	rvec_top = None
	rvec_side = None
	rmat_top = None
	rmat_side = None

	top_homogeneous = np.eye(4)
	side_homogeneous = np.eye(4)

	squareSize = 5.5
	pattern_points = np.zeros((np.prod(boardSize), 3), np.float32)
	pattern_points[:, :2] = np.indices(boardSize).T.reshape(-1, 2)
	pattern_points *= squareSize

	start_vec = np.array([0,0,0,1]).reshape(-1,1)

	while True:
		if TOP:
			ret, top_frame = top.read()
			top_frame = cv2.cvtColor(top_frame, cv2.COLOR_BGR2GRAY)
			found_top, corners_top = cv2.findChessboardCorners(top_frame, boardSize)
			vis_top = cv2.cvtColor(top_frame, cv2.COLOR_GRAY2BGR)
			if found_top:
				cv2.drawChessboardCorners(vis_top, boardSize, corners_top, found_top)
				ret, rvec_top, tvec_top = cv2.solvePnP(pattern_points, corners_top, top_camera_matrix, top_dist_coefs)
				rmat_top, jac = cv2.Rodrigues(rvec_top)

				top_homogeneous[:3,:3] = rmat_top
				top_homogeneous[:3,3:4] = tvec_top.reshape(3,-1)
			cv2.imshow("top", vis_top)
			print('t_top', tvec_top)
			print('homo_top', top_homogeneous)

		if SIDE:
			ret, side_frame = side.read()
			side_frame = cv2.cvtColor(side_frame, cv2.COLOR_BGR2GRAY)
			found_side, corners_side = cv2.findChessboardCorners(side_frame, boardSize)
			vis_side = cv2.cvtColor(side_frame, cv2.COLOR_GRAY2BGR)
			if found_side:
				cv2.drawChessboardCorners(vis_side, boardSize, corners_side, found_side)
				ret, rvec_side, tvec_side = cv2.solvePnP(pattern_points, corners_side, side_camera_matrix, side_dist_coefs)
				rmat_side, jac = cv2.Rodrigues(rvec_side)

				side_homogeneous[:3,:3] = rmat_side
				side_homogeneous[:3,3:4] = tvec_side.reshape(3,-1)
			cv2.imshow("side", vis_side)
			print('t_side', tvec_side)
			print('homo_side', side_homogeneous)




		
		if TOP and SIDE:
			product = np.dot(top_homogeneous,start_vec)
			# product = np.transpose(start_vec)*top_homogeneous
			# print(start_vec)
			# print('translation', product)

			cumulative = np.dot(side_homogeneous, np.linalg.inv(top_homogeneous))
			# cum_translation = np.dot(cumulative, start_vec)
			# print('cumulative', cumulative)
			# print('cum trans', cum_translation/cum_translation[3,:])

			R = cumulative[:3,:3]
			T = cumulative[:3,3:4]

			# F = cv2.findFundamentalMatrix(corners_top, corners_side)


		if cv2.waitKey(1) & 0xFF == ord('t'):
			if TOP:
				filename1 = IMG_PATH + 'top' + str(count) + '.jpg'
				cv2.imwrite(filename1, top_frame)
				print('Saved still as: ' + filename1)

			if SIDE:
				filename2 = IMG_PATH + 'side' + str(count) + '.jpg'
				cv2.imwrite(filename2, side_frame)
				print('Saved still as: ' + filename2)
			count = count + 1
			# np.savez_compressed('camera_transform.npz',R=R, T=T)
			# print("Saved transform!")

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	top.release()
	side.release()


if __name__ == '__main__':
	main()