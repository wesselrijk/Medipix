#!/usr/bin/env python
import glob
from pandas import DataFrame
from matplotlib import pyplot as plt

filenames = glob.glob("*.txt")
plt.figure()

for i in range(len(filenames)):
    df = DataFrame.from_csv(filenames[i], sep="\t")
    plt.plot(df)

    # for row in df.itertuples():
    #     print(row[0:])

plt.show()
