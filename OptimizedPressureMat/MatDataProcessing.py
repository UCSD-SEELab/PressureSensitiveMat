#import for linear algebra
import numpy as np

#define constants
THRESHOLD = 0
NUM_ROWS = 16
NUM_COLS = 32

#adjust data to known bias in sensor location
def calc_data_baseline(data_file):
    data = np.load(data_file)
    means = np.zeroes((NUM_ROWS,NUM_COLS))
    stdevs = np.zeroes((NUM_ROWS,NUM_COLS))
    for row in range(NUM_ROWS):
        for column in range(NUM_COLS):
            means[row,column] = np.mean(data[row,column,:])
            stdevs[row,column] = np.std(data[row,column,:])
    np.save('signalBackgroundMeans.npy',means)
    np.save('signalBackgroundStdevs.npy',stdevs)



#if pressure is less than mean + (standard dev * sigma) then set to 0
def remove_background(data,frames,means,stdevs,sigma):
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            for frame in range(frames):
                if data[row,col,frame] < (means[row,col] + (sigma * stdevs[row,col])):
                    data[row,col,frame] = 0


#smooth out noisy mat signal with low-pass filter
def remove_noise(data):

#sum up all the pressure on the mat
def total_pressure(data,frame):
    sum = 0
    for row in range(16):
        for column in range(32):
            sum+=data[row,column,frame]
    return sum

