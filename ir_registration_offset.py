import numpy as np

def normalize(input):
    return np.array(input / np.linalg.norm(input))


data_optitrack = np.genfromtxt('/media/jgschornak/Data/2017-10-30 Needle Tip Tracking/Global Frame Offset.csv', delimiter=',', skip_header=7)

measurements_optitrack = []

axis_x = np.array([1,0,0]).reshape((1, 3))
axis_y = np.array([0,1,0]).reshape((1, 3))
axis_z = np.array([0,0,1]).reshape((1, 3))

for row in range(0, data_optitrack.shape[0]):
    time = data_optitrack[row, 1]

    position_marker_a = np.array(data_optitrack[row, 10:13]).reshape((1, 3))
    position_marker_b = np.array(data_optitrack[row,14:17]).reshape((1,3))
    position_marker_c = np.array(data_optitrack[row, 18:21]).reshape((1, 3))

    checkerboard_axis_x = normalize(position_marker_b - position_marker_c).reshape((3,1))
    checkerboard_axis_y = normalize(position_marker_b - position_marker_a).reshape((3,1))
    checkerboard_axis_z = normalize(np.cross(checkerboard_axis_x.ravel(), checkerboard_axis_y.ravel())).reshape((3,1))

    rotation = np.concatenate((checkerboard_axis_x, checkerboard_axis_y, checkerboard_axis_z), axis=1)
    homogeneous = np.concatenate((np.concatenate((rotation, position_marker_b.reshape((3,1))), axis=1), np.array([0,0,0,1]).reshape((1,4))), axis=0)
    measurements_optitrack.append((time, homogeneous))
    # print(homogeneous)
# print ('\n\n\n')

data_checkerboard = np.load('./data/transforms_OPTITRACK_KEEP_ME.npz')
transforms_checkerboard = data_checkerboard['transforms']
times_checkerboard = data_checkerboard['times']

measurements_checkerboard = []
for index in range(0, transforms_checkerboard.shape[0]):
    measurements_checkerboard.append((times_checkerboard[index], transforms_checkerboard[index]))

transforms = []
for measurement_optitrack in measurements_optitrack:
    # for each optitrack measurement, find the checkerboard measurement taken at nearly the same time
    measurement_checkerboard = min(measurements_checkerboard, key=lambda tup:abs(tup[0]-measurement_optitrack[0]))
    transforms.append(np.dot(measurement_checkerboard[1], measurement_optitrack[1]))
transforms_array = np.array(transforms)
# print(transforms_array)
transform_mean = np.concatenate((normalize(np.mean(transforms_array[:,:,0], axis=0)).reshape((4,1)),
                normalize(np.mean(transforms_array[:,:,1], axis=0)).reshape((4,1)),
                normalize(np.mean(transforms_array[:,:,2], axis=0)).reshape((4,1)),
                np.mean(transforms_array[:,:,3], axis=0).reshape((4,1))), axis=1)
print(transform_mean)



