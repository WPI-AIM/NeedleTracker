import numpy as np
from pyquaternion import Quaternion

data_optitrack = np.genfromtxt('/media/jgschornak/Data/2017-10-30 Needle Tip Tracking/Global Frame Offset.csv', delimiter=',', skip_header=7)

measurements_optitrack = []
for row in range(0, data_optitrack.shape[0]):
    # print "row"
    time = data_optitrack[row, 1]
    rotation = Quaternion(data_optitrack[row, 5], data_optitrack[row, 2], data_optitrack[row, 3], data_optitrack[row, 4]).rotation_matrix
    position = np.array(data_optitrack[row, 6:9]).reshape(((3, 1)))
    homogeneous = np.concatenate((np.concatenate((rotation, position), axis=1), np.array([0,0,0,1]).reshape((1,4))), axis=0)
    # print(homogeneous)
    measurements_optitrack.append((time, homogeneous))

# print measurements_optitrack
# print('\n\n\n')
# Quaternion.

data_checkerboard = np.load('./data/transforms_OPTITRACK_KEEP_ME.npz')
transforms_checkerboard = data_checkerboard['transforms']
times_checkerboard = data_checkerboard['times']

measurements_checkerboard = []
for index in range(0, transforms_checkerboard.shape[0]):
    measurements_checkerboard.append((times_checkerboard[index], transforms_checkerboard[index]))

time_max_optitrack = measurements_optitrack[-1][0]
time_max_checkerboard = measurements_checkerboard[-1][0]
print(time_max_optitrack, time_max_checkerboard)
# print len(measurements_optitrack)
# print measurements_checkerboard[212][0]

for measurement_optitrack in measurements_optitrack:
    measurement_checkerboard = min(measurements_checkerboard, key=lambda tup:abs(tup[0]-measurement_optitrack[0]))
    print measurement_checkerboard, measurement_optitrack
    print np.dot(measurement_checkerboard[1], measurement_optitrack[1].T)

    