import cv2
import numpy as np
from glob import glob
import math

import matplotlib.pyplot as plt
import struct
from mpl_toolkits.mplot3d import Axes3D

def StereoCalib(imageList, boardSize, squareSize, displayCorners = True, useCalibrated = True, showRectified = True):
	nImages = int(len(imageList)/2)
	goodImageList = []
	imageSize = None

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	topImagePoints = []
	sideImagePoints = []


	imagePoints = []
	imagePoints.append([])
	imagePoints.append([])

	for n in range(0, nImages):
		imagePoints[0].append(None)
		imagePoints[1].append(None)

	pattern_points = np.zeros((np.prod(boardSize), 3), np.float32)
	pattern_points[:, :2] = np.indices(boardSize).T.reshape(-1, 2)
	pattern_points *= squareSize
	objectPoints = []

	j = 0

	tempTop = None

	for i in range(0, nImages):
		for k in range(0,2):
			filename = imageList[i*2+k]
			print('processsing %s... ' % filename)
			img = cv2.imread(filename, 0)
			if img is None:
				break
			if imageSize is None:
				imageSize = img.shape[:2]

			h, w = img.shape[:2]
			found, corners = cv2.findChessboardCorners(img, boardSize)

			if not found:
				print('chessboard not found')
				break
			else:
				term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
				cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
				if displayCorners is True:
					# print(filename)
					vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
					cv2.drawChessboardCorners(vis, boardSize, corners, found)
					cv2.imshow("corners", vis)
					# cv2.waitKey(50)
				if k is 1:
					# Image from side camera
					goodImageList.append(imageList[i*2])
					goodImageList.append(imageList[i*2+1])
					j = j + 1
					sideImagePoints.append(corners.reshape(-1,2))
					topImagePoints.append(tempTop)
					objectPoints.append(pattern_points)
					print('Added left and right points')

				else:
					# Image from top camera
					# rightImagePoints.append(corners.reshape(-1,2))
					tempTop= corners.reshape(-1,2)
								
					# imagePoints[k].append(corners.reshape(-1,2))
					# objectPoints.append(pattern_points)
				print('OK')
		
			# print(corners)


	print(str(j) + " chessboard pairs have been detected\n")

	nImages = j
	if nImages < 2:
		print("Too few pairs to run calibration\n")
		return

	# print(imagePoints[1])
	# print(objectPoints)

	print("Img count: " + str(len(topImagePoints)))
	print("Obj count: " + str(len(objectPoints)))

	# print(np.array(imagePoints[0]))

	top_calibration = np.load('top_calibration.npz')
	side_calibration = np.load('side_calibration.npz')

	top_rms = top_calibration['rms']
	top_camera_matrix = top_calibration['camera_matrix']
	top_dist_coefs = top_calibration['dist_coefs']

	side_rms = side_calibration['rms']
	side_camera_matrix = side_calibration['camera_matrix']
	side_dist_coefs = side_calibration['dist_coefs']

	# camera_transform = np.load('camera_transform.npz')
	# R = camera_transform['R']
	# T = camera_transform['T']

	# print('Calibrating top...')
	# top_rms, top_camera_matrix, top_dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, topImagePoints, imageSize, None, None)
	print("Top Camera\nRMS:" + str(top_rms))
	print("camera matrix: " + str(top_camera_matrix))
	print("distortion coefficients: " + str(top_dist_coefs.ravel()))

	# print('Calibrating side...')
	# side_rms, side_camera_matrix, side_dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, sideImagePoints, imageSize, None, None)
	print("Side Camera\nRMS:", side_rms)
	print("camera matrix:\n", side_camera_matrix)
	print("distortion coefficients: ", side_dist_coefs.ravel())
	
	# top_undistorted = cv2.undistort(cv2.imread(imageList[0]), top_camera_matrix, top_dist_coefs)

	# cv2.imshow("Top Undistorted", top_undistorted)

	# side_undistorted = cv2.undistort(cv2.imread(imageList[1]), side_camera_matrix, side_dist_coefs)

	# cv2.imshow("Side Undistorted", side_undistorted)

	# cameraMatrix[0] = cv2.initCameraMatrix2D(np.array(objectPoints), np.array(imagePoints[0]), imageSize, 0)
	# cameraMatrix[1] = cv2.initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0)
	# print(objectPoints[0])

	# ret, rvec_top, tvec_top = cv2.solvePnP(objectPoints[0], topImagePoints[0], top_camera_matrix, top_dist_coefs)

	# ret, rvec_side, tvec_side = cv2.solvePnP(objectPoints[0], sideImagePoints[0], side_camera_matrix, side_dist_coefs)

	# print('\n')
	# print('rvec top', rvec_top)
	# print('tvec top', tvec_top)

	# print('rvec side', rvec_side)
	# print('tvec side', tvec_side)

	# print(topImagePoints[0])

	

	print('\n')
	print('Stereo calibration (cv2)...')
	retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objectPoints, topImagePoints, sideImagePoints, top_camera_matrix, top_dist_coefs, side_camera_matrix, side_dist_coefs, imageSize, (cv2.CALIB_FIX_INTRINSIC))
	print("Rotation: ", R)
	r_vec, jac = cv2.Rodrigues(R)
	print("R_vec: ", np.multiply(r_vec, 180/math.pi))
	print("Translation: ", T)
	print("Essential: ", E)
	print("Fundamental: ", F)

	print('Stereo calibration (DIY)...')
	topImagePointsConcat = topImagePoints[0]
	sideImagePointsConcat = sideImagePoints[0]
	for i in range(1, len(topImagePoints)):
		topImagePointsConcat = np.concatenate((topImagePointsConcat, topImagePoints[i]))
		sideImagePointsConcat = np.concatenate((sideImagePointsConcat, sideImagePoints[i]))


	m1 = np.ones((len(topImagePointsConcat), 3))
	m1[:,0:2] = topImagePointsConcat

	m2 = np.ones((len(sideImagePointsConcat), 3))
	m2[:,0:2] = sideImagePointsConcat


	x1, T1 = normalizePoints(m1)
	x2, T2 = normalizePoints(m2)
	# print('Normalized', x1, T)
	# Normalization matrix
	# N = np.array([[2.0/w,0.0,-1.0],[0.0,2.0/h,-1.0],[0.0,0.0,1.0]], np.float64)
	# print('N', N)
	# x1 = np.dot(N,m1.T).T
	# print('x1,',x1)
	# x2 = np.dot(N,m2.T).T
	# print('x2',x2)


	A = np.ones((len(topImagePointsConcat),9))
	A[:,0] = np.multiply(x1[:,0],x2[:,0])
	A[:,1] = np.multiply(x1[:,1],x2[:,0])
	A[:,2] = x2[:,0]
	A[:,3] = np.multiply(x1[:,0],x2[:,1])
	A[:,4] = np.multiply(x1[:,1],x2[:,1])
	A[:,5] = x2[:,1]
	A[:,6] = x1[:,0]
	A[:,7] = x1[:,1]
	# A[:,0] = np.multiply(x2[:,0],x1[:,0])
	# A[:,1] = np.multiply(x2[:,0],x1[:,1])
	# A[:,2] = x2[:,0]
	# A[:,3] = np.multiply(x2[:,1],x1[:,0])
	# A[:,4] = np.multiply(x2[:,1],x1[:,1])
	# A[:,5] = x2[:,1]
	# A[:,6] = x1[:,0]
	# A[:,7] = x1[:,1]
	print(A)

	U, D, V = np.linalg.svd(A)
	# print('U',U)
	# print('D',D)
	# print('V',V)

	V = V.conj().T
	F_new = V[:,8].reshape(3,3).copy()
	# make rank 2 
	U, D, V = np.linalg.svd(F_new);
	# print('U',U)
	# print('D',D)
	# print('V',V)

	D_diag = np.diag([D[0], D[1], 0])
	F_new = np.dot(np.dot(U, D_diag), V)

	# F_new=np.dot(N.T,np.dot(F_new,N))
	F_new = np.dot(np.dot(T2.T, F_new), T1)

	print(F_new)
#
	R, jac = cv2.Rodrigues(np.dot(np.array([[-90],[0],[0]], dtype = np.float64), math.pi/180))
	T = np.array([[0], [-130], [130]], dtype=np.float64)
	print("Rotation: ", R)
	# r_vec, jac = cv2.Rodrigues(R)
	# print("R_vec: ", r_vec)
	print("Translation: ", T)
	# print("Fundamental: ", F_new)
	# F = F_new

	# top_undistorted = cv2.undistort(cv2.imread(imageList[0]), cameraMatrix1, distCoeffs1)

	# cv2.imshow("Top Undistorted", top_undistorted)

	# side_undistorted = cv2.undistort(cv2.imread(imageList[1]), cameraMatrix2, distCoeffs2)

	# cv2.imshow("Side Undistorted", side_undistorted)

	# R1, R2, P1, P2, Q, ret1, ret2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T)

	# print('R', R)
	# R_vec, jac = cv2.Rodrigues(R)
	# print('R vec', R_vec)
	# print('T', T)
	print('\n')
	# print('Stereo rectification (cv2)...')
	# R1, R2, P1, P2, Q, ret1, ret2 = cv2.stereoRectify(top_camera_matrix, top_dist_coefs, side_camera_matrix, side_dist_coefs, imageSize, R, T, alpha=1)

	# print("R1: ", R1)
	# R1_vec, jac = cv2.Rodrigues(R1)
	# print("R1 vec: ", R1_vec)
	# print("R2: ", R2)
	# R2_vec, jac = cv2.Rodrigues(R2)
	# print("P1: ", P1)
	# print("P2: ", P2)


	# print('Q: ', Q)
	print('Stereo rectification (DIY)...')
	P1 = np.concatenate((np.dot(side_camera_matrix,np.eye(3)),np.dot(side_camera_matrix,np.zeros((3,1)))), axis = 1)
	P2 = np.concatenate((np.dot(side_camera_matrix,R),np.dot(side_camera_matrix,T)), axis = 1)
	# print("R2 vec: ", R2_vec)
	print("P1: ", P1)
	print("P2: ", P2)


	# np.savez_compressed('calibration.npz', R1=R1, R2=R2, P1=P1, P2=P2, CameraMatrix1=cameraMatrix1, CameraMatrix2=cameraMatrix2, DistCoeffs1=distCoeffs1, DistCoeffs2=distCoeffs2,R=R,T=T,E=E,F=F)
	# np.savez_compressed('calibration.npz', CameraMatrix1=top_camera_matrix, CameraMatrix2=side_camera_matrix, DistCoeffs1=top_dist_coefs, DistCoeffs2=side_dist_coefs)
	np.savez_compressed('calibration.npz', P1=P1, P2=P2, CameraMatrix1=top_camera_matrix, CameraMatrix2=side_camera_matrix, DistCoeffs1=top_dist_coefs, DistCoeffs2=side_dist_coefs,R=R,T=T,E=E,F=F)


	# path = np.load("path.npz")
	# top_path = path["top_path"]
	# side_path = path["side_path"]

	# tip3D_homogeneous = cv2.triangulatePoints(P1, P2, top_path.reshape(2,-1)[:,50:75], side_path.reshape(2,-1)[:,50:75])
	# tip3D = (tip3D_homogeneous/tip3D_homogeneous[3])[0:3]

	# # print("homogeneous coords: " , tip3D_homogeneous)

	# print("3D coords: ", tip3D)

	# ax.scatter(np.array(tip3D)[0,:],np.array(tip3D)[1,:],np.array(tip3D)[2,:])
	# # plt.show()

	# leftInputPoints = np.array(leftImagePoints[0]).reshape(2,-1)
	# rightInputPoints = np.array(rightImagePoints[0]).reshape(2,-1)

	# np.savez_compressed('points.npz',left=leftInputPoints,right=rightInputPoints)

	# print("Left inputs: " + str(leftInputPoints))

	# points = cv2.triangulatePoints(P1, P2, leftInputPoints[:,50:100], rightInputPoints[:,50:100])	

	# print('\n')
	# testPoint = points[:,0]
	# testPoint3D = testPoint/testPoint[3]

	# point3D = points/points[3,:]
	# print("3D points: " +  str(point3D))

def main():
	size = (9, 7)
	squareSize = 6 # millimeters
	sourcePath = '/home/jgschornak/NeedleGuidance/images_converging_cams/'

	top_img_mask = sourcePath + 'top*.jpg'
	top_img_names = glob(top_img_mask)

	side_img_mask = sourcePath + 'side*.jpg'
	side_img_names = glob(side_img_mask)
	# print(left_img_names)
	# print('\n')
	# print(right_img_names)
	numPairs = len(top_img_names)

	imgList = []
	for i in range(0, numPairs):

		imgList.append(sourcePath + 'top%i' % i + '.jpg')
		imgList.append(sourcePath + 'side%i' % i + '.jpg')

	print(imgList)
	
	StereoCalib(imgList, size, squareSize)

	# while True:
	# 	if cv2.waitKey(1) & 0xFF == ord('q'):
	# 		break

def normalizePoints(pts):
	centroid = np.mean(pts, axis=0)
	# print('Centroid', centroid)

	new_pts = np.array(pts - centroid)
	# print('new_pts', new_pts)

	mean_dist = np.mean(np.linalg.norm(new_pts, axis=1))
	# print('mean dist', mean_dist)

	scale = math.sqrt(2)/mean_dist

	T = np.eye(3)
	T[0,0] = scale
	T[1,1] = scale
	T[0,2] = -scale*centroid[0]
	T[1,2] = -scale*centroid[1]
	print(T)

	return np.dot(T, pts.T).T, T

if __name__ == '__main__':
	main()

