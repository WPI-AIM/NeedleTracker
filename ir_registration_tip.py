import numpy as np
from pyquaternion import Quaternion

def normalize(input):
    return np.array(input / np.linalg.norm(input))

data_optitrack_needle = np.genfromtxt('/media/jgschornak/Data/2017-10-30 Needle Tip Tracking/Needle Tip Offset.csv', delimiter=',', skip_header=7)

transforms_homogeneous = []
for row in range(0, data_optitrack_needle.shape[0]):
    rotation_body_needle = Quaternion(data_optitrack_needle[row, 25], data_optitrack_needle[row, 22], data_optitrack_needle[row, 23], data_optitrack_needle[row, 24]).rotation_matrix
    position_body_needle = np.array(data_optitrack_needle[row, 26:29]).reshape((3, 1))

    position_marker_tip = np.array(data_optitrack_needle[row, 51:54]).reshape((3, 1))
    translation_marker_tip = position_marker_tip - position_body_needle

    # rotation = np.concatenate((checkerboard_axis_x, checkerboard_axis_y, checkerboard_axis_z), axis=1)
    homogeneous = np.concatenate((np.concatenate((rotation_body_needle, translation_marker_tip.reshape((3,1))), axis=1), np.array([0,0,0,1]).reshape((1,4))), axis=0)
    transforms_homogeneous.append(homogeneous)
    print(homogeneous)

transforms_array = np.array(transforms_homogeneous)
transform_mean = np.concatenate((normalize(np.mean(transforms_array[:,:,0], axis=0)).reshape((4,1)),
                normalize(np.mean(transforms_array[:,:,1], axis=0)).reshape((4,1)),
                normalize(np.mean(transforms_array[:,:,2], axis=0)).reshape((4,1)),
                np.mean(transforms_array[:,:,3], axis=0).reshape((4,1))), axis=1)
print('\n')
print(transform_mean)
np.savez_compressed("./data/transform_tip.npz", transform=transforms_homogeneous[-1])