import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import scipy
import datetime

# program that will plot 1 pixel: see almost at the bottom for the pixel position (x,y)
# ======================================================================================================================
# User input
location_folder = '/home/wessel/Medipix/Thesis/Robbert/Measurements/200V/SPM_eq/SPM_measurement/HGM/Zr_15000_frames/'
skip_number_of_inputs = 10
# ======================================================================================================================


start = datetime.datetime.now()

# ======================================================================================================================
#  This checks which THL values are used during the experiment
THL_values = []
for f in os.listdir(location_folder):
    if f[-5:] == '0.txt':
        try:
            THL_values.append(float(f[:-10]))
        except:
            a = 1
# ======================================================================================================================


# ======================================================================================================================
# This determines the amount of pixels of the chip used
THL_values = sorted(THL_values, key=float)
f = open(location_folder + str(int(THL_values[0])) + '_thl-0' + '.txt', 'r')
for line in f:
    pixels = len(line.split())
    break
f.close()
print('Pixels of the detector:', pixels, '*', pixels)


# ======================================================================================================================


def gaussian(x, a, b, c):
    # Just one gaussian distribution
    return a * np.exp(-1 * (x - b) ** 2. / (2. * c ** 2))


def gaussian_2(x, a, b, c, d, e, f):
    # Two gaussians
    return a * np.exp(-1 * (x - b) ** 2. / (2. * c ** 2)) + d * np.exp(-1 * (x - e) ** 2. / (2. * f ** 2))


def gaussian_first(x, a, b, c):
    # variable gaussian with one fixed second gaussian
    return a * np.exp(-1 * (x - b) ** 2. / (2. * c ** 2)) + popt1[0] * np.exp(
        -1 * (x - popt1[1]) ** 2. / (2. * popt1[2] ** 2))


def get_x_fit(x_min, x_max):
    # Returns a list that is in the range of x_min to x_max
    return np.array([i for i in np.arange(x_min, x_max + 1)])


def find_nearest(array, value):
    # finds the nearest value in the array to the value value
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_max(list):
    # finds the max in a list
    m = np.amax(list)
    return np.array([i for i, j in enumerate(list) if j == m][0])


def get_files(location):
    # This function make a grid for the individual pixels with the
    # count per THL
    hit_grid = []
    for number in THL_values:
        if number % 1 == 0:
            # Makes the float numbers of the THL to int, as the file name
            # contains the interger numbers
            number = int(number)
        f = open(location + str(number) + '_thl-0' + '.txt', 'r')  # opens the file
        file_read = []
        x_pos = -1
        for line in f:
            # will read every line in the txt file
            x_pos += 1
            file_read.append([])
            for value in line.split():
                # gets the individual values in one line and adds them to the
                # individual pixel lists
                file_read[x_pos].append(float(value))
        if number == THL_values[0]:
            # if it is the first file that is read, all of the list must be
            # added to hit_grid such that it becomes a pixel*pixels list
            for x in np.arange(0, pixels):
                hit_grid.append([])
                for y in np.arange(0, pixels):
                    hit_grid[x].append([])
                    hit_grid[x][y].append(file_read[x][y])
        else:
            # if it is not the first file that is read you can just
            # add the values to the existing lists
            for x in np.arange(0, pixels):
                for y in np.arange(0, pixels):
                    hit_grid[x][y].append(file_read[x][y])
    return hit_grid


def do_fit(list, x_list, percentage):
    if percentage > 95:
        print('Please keep te percentage below 95, best is at around 20')
        return
    x = np.array([i for i in range(0, len(list))])
    list = np.array([float(i) for i in list])
    x_min = x_list[np.argmax(list[skip_number_of_inputs:])] + skip_number_of_inputs
    max_height = list[np.argmax(list[skip_number_of_inputs:]) + skip_number_of_inputs]
    peak_pos_in_list = np.argmax(list[skip_number_of_inputs:]) + skip_number_of_inputs
    if peak_pos_in_list < 20:
        max_height = list[np.argmax(list[20:]) + 20]
        peak_pos_in_list = np.argmax(list[20:]) + 20
    nearest_x = find_nearest(np.array(x_list), x_min - x_min * 0 * .01)
    i, = np.where(np.array(x_list) == nearest_x)
    results_fit = np.array(list)[i[0]:]
    x_fit = x_list[i[0]:]

    for g in range(-len(list) + 1, -peak_pos_in_list - 1):
        if list[-g] > 0.4 * max_height:
            break
    g = -g

    g += THL_values[0]

    global popt1
    global mishit
    popt1 = [0, 0, 0, 0, 0, 0]

    # ==================================================================================================================
    min_height_2 = 100
    start_pos_1 = 50
    start_pos_2 = 55
    min_pos_1_1 = 40
    min_pos_1_2 = 50
    max_pos_1_1 = 110
    min_height_1 = 0
    max_sigma_2 = 3.5
    min_sigma_2 = 1
    max_sigma_1 = 8
    # ==================================================================================================================
    try:
        popt2, pcov = curve_fit(gaussian_2, x_fit, np.array(results_fit), p0=[10000, start_pos_1, 5, 1000, float(g), 2], bounds=([min_height_1, min_pos_1_1, 0., min_height_2, float(g), min_sigma_2], [np.inf, max_pos_1_1, max_sigma_1, np.inf, float(g+20), max_sigma_2]))# , absolute_sigma=False, sigma = weight)
    except:
        try:
            popt2, pcov = curve_fit(gaussian_2, x_fit, np.array(results_fit), p0=[10000, start_pos_2, 5, 1000, float(g - 0), 2], bounds=([min_height_1, min_pos_1_2, 0., min_height_2, float(g - 0), min_sigma_2], [np.inf, max_pos_1_1, max_sigma_1, np.inf, float(g + 10), max_sigma_2]))  # , absolute_sigma=False, sigma = weight)
        except:
            try:
                nearest_x = find_nearest(np.array(x_list), x_min - x_min * 0 * .01)
                i, = np.where(np.array(x_list) == nearest_x)
                results_fit = np.array(list)[i[0]:]
                x_fit = x_list[i[0]:]
                popt2, pcov = curve_fit(gaussian_2, x_fit, np.array(results_fit), p0=[10000, start_pos_2, 5, 1000, float(g), 2], bounds=([min_height_1, min_pos_1_2, 0., min_height_2, float(g), min_sigma_2], [np.inf, max_pos_1_1, max_sigma_1, np.inf, float(g+10), max_sigma_2]))# , absolute_sigma=False, sigma = weight)
            except:
                try:
                    popt2, pcov = curve_fit(gaussian_2, x_fit, np.array(results_fit), p0=[10000, start_pos_2, 5, 1000, float(g - 0), 2], bounds=([min_height_1, min_pos_1_2, 0., min_height_2, float(g - 0), min_sigma_2], [np.inf, max_pos_1_1, max_sigma_1, np.inf, float(g + 10), max_sigma_2]))
                except:
                    try:
                        nearest_x = find_nearest(np.array(x_list), x_min - x_min * -0 * .01)
                        i, = np.where(np.array(x_list) == nearest_x)
                        results_fit = np.array(list)[i[0]:]
                        x_fit = x_list[i[0]:]
                        popt2, pcov = curve_fit(gaussian_2, x_fit, np.array(results_fit), p0=[10000, start_pos_2, 5, 1000, float(g), 1], bounds=([min_height_1, min_pos_1_2, 0., min_height_2, float(g), min_sigma_2], [np.inf, max_pos_1_1, max_sigma_1, np.inf, float(g + 10), max_sigma_2]))  # , absolute_sigma=False, sigma = weight)
                    except:
                        popt2 = [1, 1, 1, 1, 1, 1]
                        mishit += 1


    for i in range(1, 10):
        if int(popt2[1]) == int(g) or int(popt2[4]) == int(g) or popt2[1] % 1 == 0:
            g += 1.
            try:
                popt2, pcov = curve_fit(gaussian_2, x_fit, np.array(results_fit), p0=[10000, start_pos_1, 5, 1000, float(g), 2], bounds=([min_height_1, min_pos_1_1, 0., min_height_2, float(g), min_sigma_2], [np.inf, max_pos_1_1, max_sigma_1, np.inf, float(g+20), max_sigma_2]))# , absolute_sigma=False, sigma = weight)
            except:
                try:
                    popt2, pcov = curve_fit(gaussian_2, x_fit, np.array(results_fit), p0=[10000, start_pos_2, 5, 1000, float(g - 0), 2], bounds=([min_height_1, min_pos_1_2, 0., min_height_2, float(g - 0), min_sigma_2], [np.inf, max_pos_1_1, max_sigma_1, np.inf, float(g + 10), max_sigma_2]))  # , absolute_sigma=False, sigma = weight)
                except:
                    try:
                        nearest_x = find_nearest(np.array(x_list), x_min - x_min * 0 * .01)
                        i, = np.where(np.array(x_list) == nearest_x)
                        results_fit = np.array(list)[i[0]:]
                        x_fit = x_list[i[0]:]
                        popt2, pcov = curve_fit(gaussian_2, x_fit, np.array(results_fit), p0=[10000, start_pos_2, 5, 1000, float(g), 2], bounds=([min_height_1, min_pos_1_2, 0., min_height_2, float(g), min_sigma_2], [np.inf, max_pos_1_1, max_sigma_1, np.inf, float(g+10), max_sigma_2]))# , absolute_sigma=False, sigma = weight)
                    except:
                        try:
                            popt2, pcov = curve_fit(gaussian_2, x_fit, np.array(results_fit), p0=[10000, start_pos_2, 5, 1000, float(g - 0), 2], bounds=([min_height_1, min_pos_1_2, 0., min_height_2, float(g - 0), min_sigma_2], [np.inf, max_pos_1_1, max_sigma_1, np.inf, float(g + 10), max_sigma_2]))
                        except:
                            try:
                                nearest_x = find_nearest(np.array(x_list), x_min - x_min * -0 * .01)
                                i, = np.where(np.array(x_list) == nearest_x)
                                results_fit = np.array(list)[i[0]:]
                                x_fit = x_list[i[0]:]
                                popt2, pcov = curve_fit(gaussian_2, x_fit, np.array(results_fit), p0=[10000, start_pos_2, 5, 1000, float(g), 1], bounds=([min_height_1, min_pos_1_2, 0., min_height_2, float(g), min_sigma_2], [np.inf, max_pos_1_1, max_sigma_1, np.inf, float(g + 10), max_sigma_2]))  # , absolute_sigma=False, sigma = weight)
                            except:
                                popt2 = [1, 1, 1, 1, 1, 1]
                                mishit += 1
        else:
            break

        
    
    
    if popt2[1] < popt2[4]:
        popt1[0], popt1[1], popt1[2], popt1[3], popt1[4], popt1[5] = popt2[3], popt2[4], popt2[5], popt2[0], popt2[1], popt2[2]
    else:
        popt1[0], popt1[1], popt1[2], popt1[3], popt1[4], popt1[5] = popt2[0], popt2[1], popt2[2], popt2[3], popt2[4], popt2[5]
        

    

    new_list = [list[i] - gaussian(x_list[i], popt1[0], popt1[1], popt1[2]) for i in range(0, len(list))]

    nearest_x = find_nearest(np.array(x_list), x_min - x_min * 5 * .01)
    i, = np.where(np.array(x_list) == nearest_x)
    results_fit_new = np.array(new_list)[i[0]:]
    x_fit = x_list[i[0]:]

    weight = []
    for j in range(0, len(results_fit_new)):
        if j >= i[0] - 1:
            if results_fit_new[j] < max_height * .2:
                weight.append(1)
            else:
                weight.append(1)
        else:
            weight.append(1)
    print(peak_pos_in_list + THL_values[0])
    try:
        popt, pcov = curve_fit(gaussian, x_fit, np.array(results_fit_new), p0=[5000, x_list[peak_pos_in_list], 5], bounds=([0., 30, 1.], [np.inf, np.inf, 10.]), absolute_sigma=False, sigma=weight)
    except:
        popt, pcov = [1, 1, 1], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        mishit += 1
    if popt[1] > popt1[1]:
        popt[0], popt[1], popt[2], popt1[0], popt1[1], popt1[2] = popt1[0], popt1[1], popt1[2], popt[0], popt[1], popt[2]
    print(float(sum(list[10:int(.8 * peak_pos_in_list)])) / len(list[10:int(.8 * peak_pos_in_list)]) / popt[0])
    return x, x_fit, results_fit, popt, pcov


def equalise(list, number):
    # will equalise the list for number number of bins next to the position
    if number == 0:
        return list
    pos = 1
    list_new = []
    for i in list:
        sum = 0
        if pos < number:
            for j in range(0, number + 1):
                sum += float(list[int(pos + j)])
                length = len(range(0, number + 1))
        else:
            if pos > len(list) - 2 * number:
                for j in range(0, len(list) - 1 - pos):
                    sum += float(list[int(pos) - j])
                    length = len(range(0, number + 1))
            else:
                for j in np.arange(-1 * number, number + 1):
                    sum += float(list[int(pos + j)])
                    length = len(np.arange(-1 * number, number + 1))
        list_new.append(sum / length)
        pos += 1
    return list_new

global mishit

# ======================================================================================================================
x = 0
y = 2
# ======================================================================================================================

mishit = 0
data = get_files(location_folder)
y_list = equalise(data[x][y], 0)
y_list = -np.diff(y_list)
x_list = np.arange(THL_values[0] + .5, THL_values[-1] + .5, 1)
print(len(data))
print(len(data[1]))
print(len(data[1][1]))
plt.figure(figsize=(12, 5))


data = do_fit(y_list, x_list, 1)
plt.plot(x_list, y_list, 'bo')
plt.xlabel('THL')
plt.ylabel('dN/dTHL')
plt.xlim([20, 90])
plt.ylim([-10, 10000])
print('d', data[3][1], popt1[1], popt1[1]/data[3][1], popt1[0], data[3][2])
print(data[3][0], data[3][1], data[3][2], popt1[0], popt1[1], popt1[2])
xfine = np.linspace(0., 200., 1200)  # define values to plot the function for

plt.plot(xfine, gaussian(xfine, data[3][0], data[3][1], data[3][2]), 'g-', label='Norm. dist. 1')
plt.plot(xfine, gaussian(xfine, popt1[0], popt1[1], popt1[2]), 'y-', label='Norm. dist. 2')
plt.plot(xfine, gaussian_2(xfine, data[3][0], data[3][1], data[3][2], popt1[0], popt1[1], popt1[2]), 'r-', label='Total fit')
plt.legend()


plt.show()


