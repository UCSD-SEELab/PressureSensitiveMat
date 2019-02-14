import numpy as np

v = np.load("pressuremat_data_subject1.npy")
from read_data import *

path = "./MQTT_Messages_subject5_12-14-18.txt"
raw_data = RawDataDigester(path)
pm = raw_data.get_pressuremat_data()

N = len(pm[0].keys())-1
K = len(pm[0]["row00"])

pm_data = np.zeros((len(pm), N, K))
pm_ts = []
ix = 0
for val in pm:
    tmp_array = np.zeros((N,K))
    for ixx in range(N):
        tmp_array[ixx,:] = val["row{:02d}".format(ixx)]
    pm_data[ix,:,:] = tmp_array
    pm_ts.append(val["timestamp"])
    ix += 1

pm_ts_array = np.array(pm_ts)
np.save("pressuremat_data_subject2.npy", pm_data)
np.save("pressuremat_timestamps_subject2.npy", pm_ts_array)
