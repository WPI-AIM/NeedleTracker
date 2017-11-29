import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def ProcessData(directory, insertion_start, insertion_end):
    trajectory2d_measured = np.genfromtxt(directory + '/trajectory2d.csv', delimiter=',')

    trajectory2d_canonical = np.genfromtxt(directory + '/trajectory2d_canonical.csv',delimiter=',')
    points_top_measured = trajectory2d_measured[:, 0:2]
    points_side_measured = trajectory2d_measured[:, 2:4]

    points_top_canonical = trajectory2d_canonical[:, 0:2]
    points_side_canonical = trajectory2d_canonical[:, 2:4]

    errors_top = points_top_measured - points_top_canonical
    errors_side = points_side_measured - points_side_canonical

    error_top_insert_mean = np.mean(errors_top[insertion_start:insertion_end, :], axis=0)
    error_top_insert_stdev = np.std(errors_top[insertion_start:insertion_end, :], axis=0)

    error_side_insert_mean = np.mean(errors_side[insertion_start:insertion_end, :], axis=0)
    error_side_insert_stdev = np.std(errors_side[insertion_start:insertion_end, :], axis=0)

    errors_top_insert = errors_top[insertion_start:insertion_end,:]
    errors_side_insert = errors_side[insertion_start:insertion_end,:]

    return (error_top_insert_mean, error_top_insert_stdev, error_side_insert_mean, error_side_insert_stdev, errors_top_insert, errors_side_insert)

trial1_directory = '/home/jgschornak/NeedleLocalization/data/validation_2d_2017_11_28_12_03_39'
trial2_directory = '/home/jgschornak/NeedleLocalization/data/validation_2d_2017_11_28_12_02_50'
trial3_directory = '/home/jgschornak/NeedleLocalization/data/validation_2d_2017_11_28_12_01_53'
trial4_directory = '/home/jgschornak/NeedleLocalization/data/validation_2d_2017_11_28_11_59_54'
trial5_directory = '/home/jgschornak/NeedleLocalization/data/validation_2d_2017_11_28_12_00_46'

trial1_result = ProcessData(trial1_directory, 7, 68)
trial2_result = ProcessData(trial2_directory, 6, 82)
trial3_result = ProcessData(trial3_directory, 65, 126)
trial4_result = ProcessData(trial4_directory, 5, 105)
trial5_result = ProcessData(trial5_directory, 10, 113)

# plt.figure(1)
# plt.hist(trial1_result[4][:,0], bins='auto')  # arguments are passed to np.histogram
# plt.title("Insertion Error (X)")
# plt.figure(2)
# plt.hist(trial1_result[4][:,1], bins='auto')  # arguments are passed to np.histogram
# plt.title("Insertion Error (Y)")


# print(trial1_result[4][:,0])

# print(len(trial1_result[4][:, 0]))

errors_total = np.concatenate((trial1_result[4], trial2_result[4], trial3_result[4]), axis=0)

# f_out = interp1d(np.linspace(0,100,100), trial1_result[4][:,0], axis=1)
# print(f_out)

errors_total_mean = np.mean(errors_total, axis=0)
errors_total_stdev = np.std(errors_total, axis=0)
# print(errors_total)
print(errors_total_mean, errors_total_stdev)

errors_total_norm = np.linalg.norm(errors_total, axis=1)
# print(errors_total_norm)
errors_total_norm_mean = np.mean(errors_total_norm, axis=0)
errors_total_norm_stdev = np.std(errors_total_norm, axis=0)
print(errors_total_norm_mean, errors_total_norm_stdev)

plt.figure(3)
plt.plot(np.linspace(0, 100, len(trial1_result[4][:, 0])), trial1_result[4][:, 0], 'r',
         np.linspace(0, 100, len(trial2_result[4][:, 0])), trial2_result[4][:, 0], 'g',
         np.linspace(0, 100, len(trial3_result[4][:, 0])), trial3_result[4][:, 0], 'b',
         # np.linspace(0, 100, len(trial4_result[4][:, 0])), trial4_result[4][:, 0], 'k',
         # np.linspace(0, 100, len(trial5_result[4][:, 0])), trial5_result[4][:, 0], 'm',
         np.linspace(0, 100, len(trial1_result[4][:, 1])), trial1_result[4][:, 1], 'r--',
         np.linspace(0, 100, len(trial2_result[4][:, 1])), trial2_result[4][:, 1], 'g--',
         np.linspace(0, 100, len(trial3_result[4][:, 1])), trial3_result[4][:, 1], 'b--',
         # np.linspace(0, 100, len(trial4_result[4][:, 1])), trial4_result[4][:, 1], 'k--',
         # np.linspace(0, 100, len(trial5_result[4][:, 1])), trial5_result[4][:, 1], 'm--'
         )
plt.legend(['Trial 1 X', 'Trial 2 X', 'Trial 3 X', 'Trial 1 Y', 'Trial 2 Y', 'Trial 3 Y'])
plt.xlabel('Percent Maximum Depth')
plt.ylabel('Error (px)')

plt.show()
