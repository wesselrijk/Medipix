#!/usr/bin/env python
import glob
from natsort import natsorted
from matplotlib import pyplot as plt
from skimage import io
import seaborn as sns
import numpy as np


def rescale_list(y_list):
    maximum = max(y_list)
    y_list = (y_list / maximum) * 100
    return y_list


filenames = natsorted(glob.glob("*thl0.tiff"))
x_list = []
count_list = []
pixel_list = []

plt.figure()

for i in range(len(filenames)-14):
    im = np.array(io.imread(filenames[i+14]))
    count = np.sum(im)
    pixel = im[0][0]
    thl = int(filenames[i][3:].partition("_")[0])
    x_list.append(thl)
    count_list.append(count)
    pixel_list.append(pixel)


rescaled_list = rescale_list(count_list)
rescaled_list = [float(i) for i in rescaled_list]
diff_list = -np.diff(rescaled_list)


sns.set(style="ticks")
#plt.plot(x_list, y_list)
plt.plot(x_list, diff_list)
plt.yscale('log')
plt.show()


#
#sns.set(style="ticks")
#filenames = glob.glob("*.tiff")
##plt.figure()
#
#plt.figure()
#
#for i in range(0):
#    df = DataFrame.from_csv(filenames[i], sep="\t")
#    plt.plot(df)
#
#    # for row in df.itertuples():
#    #     print(row[0:])
#
#plt.show()
