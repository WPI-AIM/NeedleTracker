import numpy as np

trajectory2d_measured = np.genfromtxt(
    '/home/jgschornak/NeedleLocalization/data/ground_truth_2017_11_27_20_39_11/trajectory2d.csv', delimiter=',')

trajectory2d_canonical = np.genfromtxt(
    '/home/jgschornak/NeedleLocalization/data/ground_truth_2017_11_27_20_39_11/trajectory2d_canonical.csv', delimiter=',')

points_top_measured = trajectory2d_measured[:,0:2]
points_side_measured = trajectory2d_measured[:,2:4]

points_top_canonical = trajectory2d_canonical[:,0:2]
points_side_canonical = trajectory2d_canonical[:,2:4]

length = len(points_top_measured)
print(length)

for index in range(0,length):
    point_top_measured = points_top_measured[index,:]
    point_side_measured = points_side_measured[index,:]

    point_top_canonical = points_top_canonical[index,:]
    point_side_canonical = points_side_canonical[index,:]
    print(point_top_measured, point_side_measured, point_top_canonical, point_side_canonical)


    error_top = np.linalg.norm(point_top_measured - point_top_canonical)
    error_side = np.linalg.norm(point_side_measured - point_side_canonical)
    print(error_top, error_side)