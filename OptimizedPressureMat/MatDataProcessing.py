# import for linear algebra
import random

import numpy as np
import matplotlib.pyplot as plt
import math
# import DBSCAN
import sklearn
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time
import seaborn as sns
import sys
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

sys.path.append('../')
print("pre-import")
from libs.MeanShift_py import mean_shift as gaussian_mean_shift
from libs.LeastSquares import ls_classifier as LLS
import xlsxwriter

print("post-import")

# define constants
THRESHOLD = 0
NUM_ROWS = 16
NUM_COLS = 32
NOISE_SIGMA = 2  # The number of stdevs within which noise must exist of the background signal
# to be considered noise, going beyond is considered active signal
DBSCAN_ESP = 2  # Radius that our DBSCAN considers for a cluster
DBSCAN_MIN_SAMPLES = 6  # minimum number of points our DBSCAN needs in a radius to consider something a cluster
MEAN_SHIFT_QUANTILE = 0.3  # [0,1] the multiplier of pairwise distances, a greater quantile means less clusters

PLOT_EN = True
PRINT_DATA_EN = False
CLUSTERING_ALGORITHMS = ['DBSCAN', 'MEAN_SHIFT']
CLASSIFIER_TYPES = ['LEAST_SQUARES', 'SVM', 'MLP', 'NEAREST_CENTROID']
CLUSTERING_ALG = CLUSTERING_ALGORITHMS[0]
GAUSSIAN_KERNEL = False
WEIGHTED_CLUSTER = True

FILES = ['./data/pressuremat_data_subject4.npy', './data/pressuremat_data_subject5.npy',
         './data/pressuremat_data_subject6.npy']

# create global variable to store features in
dataFeatures = list()


# adjust data to known bias in sensor location
def calc_data_baseline(data, meansList, stdevsList):
    # data = np.load(data_file)
    means = np.zeros((NUM_ROWS, NUM_COLS))
    stdevs = np.zeros((NUM_ROWS, NUM_COLS))
    for row in range(NUM_ROWS):
        for column in range(NUM_COLS):
            means[row, column] = np.mean(data[:, row, column])
            stdevs[row, column] = np.std(data[:, row, column])
    np.save('signalBackgroundMeans.npy', means)
    meansList = means
    np.save('signalBackgroundStdevs.npy', stdevs)
    stdevsList = stdevs
    return


# if pressure is less than mean + (standard dev * sigma) then set to 0
def remove_background(data, frames, means, stdevs, sigma):
    originalData = np.copy(data)
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            for frame in range(frames):
                # if data[frame, row, col] < (means[row, col] + (sigma * stdevs[row, col])):
                #    data[frame, row, col] = 0
                if data[frame, row, col] < 1400:
                    data[frame, row, col] = 0
    return originalData


# smooth out noisy mat signal with low-pass filter
# DBSCAN deals better with noise than other clustering algorithms though, so I am saving the
# Implementation for later
def remove_noise(data):
    return


# parse a data set for non-zero values and add their x,y values as a pair into a new numpy array
# data must be a 2 dimensional numpy array
def grab_active_x_y(data):
    activeList = [[]]
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            if data[row, col] > 0:
                activeList.append([row, col])
    activeList = [x for x in activeList if x != []]
    activeNPArr = np.asarray(activeList)

    return activeNPArr


# sum up all the pressure on the mat and frame is the pressure values in those points
# the number of clusters is the number of clusters on the mat
def total_pressure(activePoints, pressureValues):
    activeLength = activePoints.shape
    totalPressure = 0
    for i in range(activeLength[0]):
        totalPressure = totalPressure + pressureValues[activePoints[i, 0], activePoints[i, 1]]
    return totalPressure


# Finds the center point for clusters returned in a DBSCAN Cluster
# It will take the weighted average based on the scale values of pressure
# db is the fitted dbscan while data is the data that the dbscan fitted
# data_weight is the weights of the data points
# DATA IS ACTIVE POINTS IN FRAME, wont be 16x32
# SATA_WEIGHT will be 16x32, it is the actual frame
def find_DBSCAN_centers(data, data_weight, db):
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # find the unique set of labels
    unique_labels = set(labels)
    # scale the weights of the data
    # will result in scaled values of the 16x32 frame
    maxVal = (data_weight.max())
    scaledData = data_weight / maxVal
    clusterCenters = np.zeros((n_clusters_, 2))

    # iterate through each of the unique labels
    count = 0
    for k in unique_labels:
        # ignore noise
        if k == -1:
            continue

        # else calculate the center value for this cluster
        class_member_mask = (labels == k)
        # get the data points only associated with the current cluster
        # definitely not 16x32, its values are coordinates in a frame
        clusterData = data[class_member_mask]
        numSamples = clusterData.shape
        averageX = 0
        averageY = 0
        weightSum = 0

        for i in range(numSamples[0]):
            # multiply scaled pressure value times the location in the frame of the active pressure
            if WEIGHTED_CLUSTER:
                # compute weighted average
                averageX += (scaledData[clusterData[i, 0], clusterData[i, 1]] * (clusterData[i, 1]))
                averageY += (scaledData[clusterData[i, 0], clusterData[i, 1]] * (clusterData[i, 0]))
                weightSum += scaledData[clusterData[i, 0], clusterData[i, 1]]

            # This works well for plotting clusters without weights
            else:
                averageX += clusterData[i, 1]
                averageY += clusterData[i, 0]
                weightSum += 1

        averageX = averageX / weightSum
        averageY = averageY / weightSum
        clusterCenters[count, 0] = averageX
        clusterCenters[count, 1] = averageY
        count = count + 1

    # print("clusterCenters: " + str(clusterCenters))
    # print("num clusters: " + str(n_clusters_))
    return clusterCenters


# create features from MataDataProcessing library
# data is the actual data we will do this from
# db is the DBSCAN that fitted the data
# numActiveFrames is however many clusters = 2 frames we
# want to use to create our feature set
# returns: two values - total pressure, distance between cluster centers
def create_features_DBSCAN(activePoints, frame, db, clusterCenters):
    from scipy.spatial import distance
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # find the unique set of labels
    unique_labels = set(labels)
    totalPressure = 0
    covariance_matrix = np.zeros((clusterCenters.shape[0], 2, 2))

    # Calculate each clusters covariance matrix
    for k in unique_labels:
        # ignore noise points
        if k == -1:
            continue
        class_member_mask = (labels == k)
        totalPressure = totalPressure + total_pressure(activePoints[class_member_mask], frame)
        covariance_matrix[k] = (calculate_covariance_matrix(activePoints[class_member_mask]))

    # calculate the distance between each cluster
    if clusterCenters.shape[0] == 2:
        clusterDistance = distance.euclidean(clusterCenters[0], clusterCenters[1])
    else:
        clusterDistance = 0


    return [totalPressure, clusterDistance, covariance_matrix]


def calculate_covariance_matrix(activePoints):
    # matrix multiply to find the cluster center (active points only)
    # 1 (1xN) * P' (Nx2)  / N
    N = activePoints.shape[0]
    ones = np.ones((1, N))
    average = np.dot(ones, activePoints)  # 1xN * Nx2 --> 1x2
    average = average / N  # --> 1x2

    # calculate covariancce matrix, (P - Pavg')(P - Pavg')' / N
    covariance = np.dot((activePoints - average).T, (activePoints - average))  # 2XN * Nx2 --> 2x2
    covariance = covariance / N
    return covariance


# get the specific rotation matrix for this application. This rotation matrix is modified to rotate a 3D matrix in
# 2 dimensions and keep the associated pressure value with each point
# rotation_angle is the angle between the the vertical and the vertical eigenvector of the foot
# rotation_matrix is the 3x3 np array of 0's which we will value fill here to meet our specific needs
def get_rotation_matrix(rotation_angle, rotation_matrix):
    # fill in the cosine values of the rotation matrix
    rotation_matrix[0, 0] = math.cos(rotation_angle)
    rotation_matrix[1, 1] = math.cos(rotation_angle)

    # adjust the matrix if the rotation must go clockwise
    if math.sin(rotation_angle) < 0:
        rotation_matrix[0, 1] = -math.sin(rotation_angle)
        rotation_matrix[1, 0] = math.sin(rotation_angle)
    else:
        rotation_matrix[0, 1] = math.sin(rotation_angle)
        rotation_matrix[1, 0] = -math.sin(rotation_angle)

    # set final value that needs to be set to keep the pressure value associated with each rotated point
    rotation_matrix[2, 2] = 1

    return rotation_matrix


# GOALS COVARIANCE MATRIX (AND EIGENVALUES), DISTANCE BETWEEN CLUSTERS, TOTAL PRESSURE, RATIO OF FOOT PRESSURE,
# TODO Eigenvalues, ratio


def analyze_data(filenames):
    """
    TODO Add function description
    """
    dataSet = list()
    backMeans = np.zeros((len(filenames), NUM_ROWS, NUM_COLS))
    backStdevs = np.zeros((len(filenames), NUM_ROWS, NUM_COLS))
    for k, file in enumerate(filenames):
        dataSet.append(np.load(file))
        data = dataSet[-1]
        # data here is 1 frame now, problem
        calc_data_baseline(data, backMeans, backStdevs)

        dataSize = data.shape
        frameCount = dataSize[0]  # check how many frames exist in this data sample
        original_data = remove_background(data, frameCount, backMeans[k], backStdevs[k], NOISE_SIGMA)
        featureSet = list()

        # Now that background is moved, lets make cluster, compute DBSCAN for each frame!
        validFrameCount = 0
        for frame in range(frameCount):
            activeInFrame = grab_active_x_y(data[frame, :, :])
            if activeInFrame.ndim == 1:
                continue

            # check the clustering algorithm we want to use for the clustering
            if CLUSTERING_ALG == 'DBSCAN':
                db = DBSCAN(eps=DBSCAN_ESP, min_samples=DBSCAN_MIN_SAMPLES).fit(activeInFrame)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_
                # Number of clusters in labels, ignoring noise if present.
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise_ = list(labels).count(-1)

                # HERE WE PARSE THE DATA FOR FEATURES WE WILL USE IN CLASSIFIERS
                if (n_clusters_ == 2):
                    clusterCenters = find_DBSCAN_centers(activeInFrame, original_data[frame, :, :], db)
                    features = create_features_DBSCAN(activeInFrame, original_data[frame, :, :], db, clusterCenters)
                    covar_eigenvalues = np.linalg.eig(features[2])
                    flat_eig = [item for sublist in covar_eigenvalues for item in sublist]
                    # TODO: First element of covarEigenvalues is the eigenvalues, this will tell you approximate foot
                    # dimensions (length, width). The second element is eigenvectors, which explain the direction that
                    # the foot is pointing. Find the max absolute value of the eigenvalues for each cluster, determine
                    # the associated eigenvector. If the eigenvalue was negative, rotate eigenvector by 180 deg. Compare
                    # angle between the two eigenvectors and use this as a feature (pidgeon-toed-ness)

                    # TODO: For eigenvalues, average to determine approximate foot size. Or if you're worried about
                    # subject being completely on mat or not, take max eigenvalue for each direction or just the overall
                    # max.

                    # find the angle between both feet
                    w, v = np.linalg.eig(features[2])
                    foot_angle = angle_between(v[0][0], v[1][0])

                    # TODO: rotate the clusters to align them vertically and then interpolate them to an equal size
                    vertical_vector = [0, 1]
                    vertical_vector = np.array(vertical_vector)
                    left_max_eig = max((flat_eig[0][0]), abs(flat_eig[0][0]))
                    right_max_eig = max((flat_eig[1][0]), abs(flat_eig[1][0]))
                    left_angle = angle_between(v[0][0], vertical_vector)
                    right_angle = angle_between(v[1][0], vertical_vector)

                    # build the rotation matrices
                    rotation_matrix_left = get_rotation_matrix(left_angle, np.zeros([3, 3]))
                    rotation_matrix_right = get_rotation_matrix(right_angle, np.zeros([3, 3]))

                    # assemble np array to properly multiply with the rotation matrices
                    # This will be a (3 x N) matrix where row 0 is xval, row 1 is yval, and row 2 is pressure val and
                    # each column is its own point
                    rotated_left = 0
                    rotated_right = 0
                    unique_labels = set(labels)
                    for point_label in unique_labels:
                        # ignore noise points
                        if point_label == -1:
                            continue
                        class_member_mask = (labels == point_label)
                        mask_size = sum(class_member_mask)
                        if point_label == 0:
                            rotated_left = np.zeros([3, mask_size])
                            rotated_left[0] = activeInFrame[class_member_mask, 0]
                            rotated_left[1] = activeInFrame[class_member_mask, 1]
                            rotated_left[2] = data[frame, activeInFrame[class_member_mask, 0], activeInFrame[class_member_mask, 1]]
                        else:
                            rotated_right = np.zeros([3, mask_size])
                            rotated_right[0] = activeInFrame[class_member_mask, 0]
                            rotated_right[1] = activeInFrame[class_member_mask, 1]
                            rotated_right[2] = data[frame, activeInFrame[class_member_mask, 0], activeInFrame[class_member_mask, 1]]

                    # we have our rotation matrix and our matrices that we need to rotate as well
                    # also normalize their locations
                    rotated_left = np.dot(rotation_matrix_left, rotated_left)
                    min_left_x = min(rotated_left[0])
                    min_left_y = min(rotated_left[1])
                    rotated_left[0] = rotated_left[0] - min_left_x
                    rotated_left[1] = rotated_left[1] - min_left_y
                    #get covariance matrix of the rotated matrix first
                    left_cov = np.cov(rotated_left[:2])
                    w_left, v_left = np.linalg.eig(left_cov)

                    rotated_right = np.dot(rotation_matrix_right, rotated_right)
                    min_right_x = min(rotated_right[0])
                    min_right_y = min(rotated_right[1])
                    rotated_right[0] = rotated_right[0] - min_right_x
                    rotated_right[1] = rotated_right[1] - min_right_y
                    #get the covariance matrix of the rotated matrix first
                    right_cov = np.cov(rotated_right[:2])
                    w_right, v_right = np.linalg.eig(right_cov)

                    #here we will interpolate the images
                    import scipy.interpolate
                    import matplotlib.pyplot as plt
                    left_cart_coord = list(zip(rotated_left[0], rotated_left[1]))
                    left_x = np.linspace(min(rotated_left[0]), max(rotated_left[0])) #can add n= to increase density
                    left_y = np.linspace(min(rotated_left[1]), max(rotated_left[1]))
                    left_x, left_y = np.meshgrid(left_x, left_y)
                    interp_left = scipy.interpolate.LinearNDInterpolator(left_cart_coord, rotated_left[2], fill_value=0)
                    left_z = interp_left(left_x, left_y)

                    right_cart_coord = list(zip(rotated_right[0], rotated_right[1]))
                    right_x = np.linspace(min(rotated_right[0]), max(rotated_right[0]))  # can add n= to increase density
                    right_y = np.linspace(min(rotated_right[1]), max(rotated_right[1]))
                    right_x, right_y = np.meshgrid(right_x, right_y)
                    interp_right = scipy.interpolate.LinearNDInterpolator(right_cart_coord, rotated_right[2], fill_value=0)
                    right_z = interp_right(right_x, right_y)

                    # Set up our feature set
                    featureSet.append([features[0], features[1], flat_eig[0][0], flat_eig[0][1], flat_eig[1][0],
                                       flat_eig[1][1], foot_angle, k])
                    validFrameCount = validFrameCount + 1

                    # print information if printing is enabled
                    if PRINT_DATA_EN:
                        # PRINT WHAT WE FOUND OUT  ABOUT EACH FRAME
                        if (n_clusters_ > 0):
                            print('Estimated number of clusters: %d' % n_clusters_)
                            print('Estimated number of noise points: %d' % n_noise_)
                        else:
                            continue

                    # PLOT RESULTS
                    if PLOT_EN:
                        import matplotlib.pyplot as plt
                        plt.subplot(2, 4, 1)

                        # Black removed and is used for noise instead.
                        unique_labels = set(labels)
                        colors = [plt.cm.Spectral(each)
                                  for each in np.linspace(0, 1, len(unique_labels))]
                        for k, col in zip(unique_labels, colors):
                            if k == -1:
                                # Black used for noise.
                                col = [0, 0, 0, 1]

                            class_member_mask = (labels == k)

                            xy = activeInFrame[class_member_mask & core_samples_mask]
                            plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col),
                                     markeredgecolor='k', markersize=14)

                            xy = activeInFrame[class_member_mask & ~core_samples_mask]
                            plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col),
                                     markeredgecolor='k', markersize=6)
                        for i in range(clusterCenters.shape[0]):
                            plt.plot(clusterCenters[i][0], clusterCenters[i][1], 'x', markeredgecolor='k',
                                     markerSize=15)
                        plt.xlim(0, 32)
                        plt.ylim(0, 20)
                        plt.title('Estimated number of clusters: %d' % n_clusters_)

                        # plot a heatmap
                        plt.subplot(2, 4, 2)
                        plt.xlim(0, 16)
                        plt.ylim(0, 32)
                        ax = sns.heatmap(original_data[frame, :, :])
                        ax.invert_yaxis()

                        # Plot the Histogram of pressure points
                        pressure_array = original_data[frame, :, :].flatten()
                        num_bins = 60
                        plt.subplot(2, 3, 3)
                        plt.hist(pressure_array, num_bins)

                        # Plot the rotated left foot
                        plt.subplot(2, 4, 4)
                        plt.plot(rotated_left[1], rotated_left[0], 'o', markerfacecolor=tuple(col),
                                 markeredgecolor='k', markersize=14)

                        plt.xlim(-2, max(rotated_left[1]) + 2)
                        plt.ylim(-2, max(rotated_left[0]) + 2)
                        plt.title("Eig Vec: " + str(round(v_left[0, 0], 3)) + " " + str(round(v_left[0, 1], 3)) + " " +
                                  str(round(v_left[1, 0], 3)) + " " + str(round(v_left[1, 1], 3)))


                        # Plot the rotated right foot
                        plt.subplot(2, 4, 5)
                        plt.plot(rotated_right[1], rotated_right[0], 'o', markerfacecolor=tuple(col),
                                 markeredgecolor='k', markersize=14)
                        plt.xlim(-2, max(rotated_right[1]) + 2)
                        plt.ylim(-2, max(rotated_right[0]) + 2)
                        plt.title("Eig Vec: " + str(round(v_right[0, 0], 3)) + " " + str(round(v_right[0, 1], 3)) + " "
                                  + str(round(v_right[1, 0], 3)) + " " + str(round(v_right[1, 1], 3)))

                        plt.subplot(2, 4, 6)
                        plt.pcolormesh(left_x, left_y, left_z)
                        plt.colorbar()

                        plt.subplot(2, 4, 7)
                        plt.pcolormesh(right_x, right_y, right_z)
                        plt.colorbar()


                        plt.draw()
                        plt.pause(1e-17)
                        time.sleep(0.1)
                        plt.clf()

            # Otherwise we will use the Mean shift algorithm for our clustering algorithm
            else:
                # take the estimated value for the bandwidth for mean shift

                # time.sleep(20)
                # bandwidth = 10

                # use a flat kernel

                if not GAUSSIAN_KERNEL:
                    bandwidth = (estimate_bandwidth(activeInFrame, quantile=MEAN_SHIFT_QUANTILE)) / 1
                    print('executing flat kernel')
                    if bandwidth == 0:
                        continue
                    ms = MeanShift(bandwidth=bandwidth).fit(activeInFrame)
                    labels = ms.labels_
                    cluster_centers = ms.cluster_centers_

                    # calculate the number of clusters
                    labels_unique = np.unique(labels)
                    n_clusters_ = len(labels_unique)

                # use a gaussian kernel
                else:
                    bandwidth = np.zeros(2)
                    bandwidth[0] = (estimate_bandwidth(activeInFrame, quantile=MEAN_SHIFT_QUANTILE)) / 1.4
                    bandwidth[1] = bandwidth[0] / 10
                    if bandwidth.any() == 0:
                        continue
                    print("executing gaussian Kernel")
                    activeShape = activeInFrame.shape
                    pressureWeights = np.zeros(activeShape[0])
                    for i in range(activeShape[0]):
                        pressureWeights[i] = data[frame, activeInFrame[i, 0], activeInFrame[i, 1]]
                    pressureWeights = scale(pressureWeights)
                    ms = gaussian_mean_shift.MeanShift(kernel='multivariate_gaussian')
                    mean_shift_result = ms.cluster(activeInFrame, bandwidth, pressureWeights)
                    cluster_centers = mean_shift_result.shifted_points
                    labels = mean_shift_result.cluster_ids

                    labels_unique = np.unique(labels)
                    n_clusters_ = len(labels_unique)

                # print information if printing is enabled
                if PRINT_DATA_EN:
                    # PRINT WHAT WE FOUND OUT  ABOUT EACH FRAME
                    if (n_clusters_ > 0):
                        print('Estimated number of clusters: %d' % n_clusters_)
                    else:
                        continue

                # if plotting is enabled then continuosly update the plot
                if PLOT_EN:
                    import matplotlib.pyplot as plt

                    plt.clf()
                    plt.subplot(2, 2, 1)

                    from itertools import cycle

                    colors = [plt.cm.Spectral(each)
                              for each in np.linspace(0, 1, len(labels_unique))]
                    for k, col in zip(range(n_clusters_), colors):
                        my_members = labels == k
                        cluster_center = cluster_centers[k]
                        plt.plot(activeInFrame[my_members, 1], activeInFrame[my_members, 0], '.',
                                 markerfacecolor=tuple(col),
                                 markeredgecolor='k', markersize=10)
                        plt.plot(cluster_center[1], cluster_center[0], 'o',
                                 markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
                    plt.xlim(-5, 32)
                    plt.ylim(-5, 20)
                    plt.title('Estimated number of clusters: %d' % n_clusters_)

                    # PLOT THE HEATMAP
                    plt.subplot(2, 2, 2)
                    plt.xlim(0, 16)
                    plt.ylim(0, 32)
                    ax = sns.heatmap(original_data[frame, :, :])
                    ax.invert_yaxis()

                    # Histogram Plot
                    # plt.subplot(2,2,3)
                    # for x in range(NUM_ROWS):
                    #    for y in range(NUM_COLS):

                    plt.draw()
                    plt.pause(1e-17)
                    time.sleep(0.01)
                    plt.clf()

        dataFeatures.append(featureSet)
        np.save('temp_features.npy', dataFeatures)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


# create a classifier to run on data stored in featureSet which is generated by analyze_data
def create_classifier(classifierType, train_ratio, hidden_layer_sizes=(100, 2), activation='relu', max_iter=5000,
                      alpha=1e-4,
                      solver='adam', verbose=0, tol=1e-6,
                      learning_rate_init=0.001, shuffle=True, exclude=None, n_components=7):
    # PREP THIS DATA
    # shuffle the valid features to make them randomly selected
    dataHistogram = list()

    npDataFeatures = np.concatenate(dataFeatures, axis=0)
    # create a mask for class 0's
    class_0_mask = npDataFeatures[:, -1] == 0

    """
    bins = np.linspace(0, 20, 50)
    plt.subplot(7, 2, 1)
    plt.style.use('seaborn-deep')
    bins = np.linspace(0, 150000, 50)
    plt.hist(npDataFeatures[class_0_mask, 0], 50, label='0')
    plt.hist(npDataFeatures[~class_0_mask, 0], 50, label='1')
    plt.title('feat 0')

    plt.subplot(7, 2, 2)
    bins = np.linspace(0, 20, 50)
    plt.hist(npDataFeatures[class_0_mask, 1], 50, label='0')
    plt.hist(npDataFeatures[~class_0_mask, 1], 50, label='1')
    plt.title('feat 1')

    plt.subplot(7, 2, 3)
    bins = np.linspace(0, 12, 50)
    plt.hist(npDataFeatures[class_0_mask, 2], 50, label='0')
    plt.hist(npDataFeatures[~class_0_mask, 2], 50, label='1')
    plt.title('feat 2')

    plt.subplot(7, 2, 4)
    plt.hist(npDataFeatures[class_0_mask, 3], 50, label='0')
    plt.hist(npDataFeatures[~class_0_mask, 3], 50, label='1')
    plt.title('feat 3')

    plt.subplot(7, 2, 5)
    plt.hist(npDataFeatures[class_0_mask, 4], 50, label='0')
    plt.hist(npDataFeatures[~class_0_mask, 4], 50,  label='1')
    plt.title('feat 4')

    plt.subplot(7, 2, 6)
    plt.hist(npDataFeatures[class_0_mask, 5], 50, label='0')
    plt.hist(npDataFeatures[~class_0_mask, 5], 50,  label='1')
    plt.title('feat 5')

    plt.subplot(7, 2, 7)
    plt.hist(npDataFeatures[class_0_mask, 6], 50,  label='0')
    plt.hist(npDataFeatures[~class_0_mask, 6], 50, label='1')
    plt.title('feat 6')
    plt.show()
    """

    feature_train_mask = np.zeros(npDataFeatures.shape[0], dtype=bool)
    feature_train_indices = random.sample(range(0, npDataFeatures.shape[0]), int(train_ratio * npDataFeatures.shape[0]))
    feature_train_mask[feature_train_indices] = True

    # Select training and test feats

    allTrainFeats = npDataFeatures[feature_train_mask, :]
    allTestFeats = npDataFeatures[~feature_train_mask, :]

    if exclude != None:
        allTrainFeats = np.delete(allTrainFeats, exclude, axis=1)
        allTestFeats = np.delete(allTestFeats, exclude, axis=1)

    # LEAST SQUARES
    if classifierType == CLASSIFIER_TYPES[0]:
        # Least squares model
        model = LLS.train(allTrainFeats[:, :-1], allTrainFeats[:, -1])
        predLabelsTrain = LLS.predict(model, allTrainFeats[:, :-1])
        predLabelsTest = LLS.predict(model, allTestFeats[:, :-1])

        print("Least Squares Classifier training")
        print("Train accuracy: {0:0.4f}".format(metrics.accuracy_score(allTrainFeats[:, -1], predLabelsTrain)))
        print("Test accuracy: {0:0.4f}\n".format(metrics.accuracy_score(allTestFeats[:, -1], predLabelsTest)))
        return metrics.accuracy_score(allTestFeats[:, -1], predLabelsTest)

    # SVM
    elif classifierType == CLASSIFIER_TYPES[1]:
        # SVC model
        print("SVM Training")
        normalizer = sklearn.preprocessing.MinMaxScaler().fit(allTrainFeats[:, :-1])
        model = svm.LinearSVC()
        model.fit(normalizer.transform(allTrainFeats[:, :-1]), allTrainFeats[:, -1])
        print('Train accuracy: {0:0.4f}'.format(
            model.score(normalizer.transform(allTrainFeats[:, :-1]), allTrainFeats[:, -1])))
        print('Test accuracy: {0:0.4f}\n'.format(
            model.score(normalizer.transform(allTestFeats[:, :-1]), allTestFeats[:, -1])))
        return model.score(normalizer.transform(allTestFeats[:, :-1]), allTestFeats[:, -1])

    # MLP
    elif classifierType == CLASSIFIER_TYPES[2]:
        # MLP Neural Network model
        print('Multi Layer Perceptron Training')
        # TODO adjust with dropout, or switch to Keras or sknn, try to do dropout somewhere
        # TODO also print out stdev, var, maybe loss, shuffle=True, mess with hidden layer sizes
        normalizer = sklearn.preprocessing.MinMaxScaler().fit(allTrainFeats[:, :-1])
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter,
                              alpha=alpha,
                              solver=solver, verbose=verbose, tol=tol,
                              learning_rate_init=learning_rate_init, shuffle=shuffle, )
        model.fit(normalizer.transform(allTrainFeats[:, :-1]), allTrainFeats[:, -1])
        print('Train accuracy: {0:0.4f}'.format(
            model.score(normalizer.transform(allTrainFeats[:, :-1]), allTrainFeats[:, -1])))
        print('Test accuracy: {0:0.4f}\n'.format(
            model.score(normalizer.transform(allTestFeats[:, :-1]), allTestFeats[:, -1])))
        return model.score(normalizer.transform(allTestFeats[:, :-1]), allTestFeats[:, -1])

    elif classifierType == CLASSIFIER_TYPES[3]:
        from sklearn.neighbors import NearestNeighbors
        allTrainFeats = sklearn.preprocessing.normalize(allTrainFeats, axis=0)
        # sklearn.preprocessing.normalize(allTestFeats, axis=0)
        model = NearestNeighbors()
        model.fit(allTrainFeats[:, :7], allTrainFeats[:, 14])
        predLabelsTest = model.predict(allTestFeats[:, :7])
        print("Nearest Centroid Training")
        print("Test accuracy: {0}".format(metrics.accuracy_score(allTestFeats[:, 14], predLabelsTest)))

    else:
        print("ERROR no classifier of that type available")
        print("Currently only options are " + str(CLASSIFIER_TYPES))
    return


if __name__ == '__main__':
    analyze_data(FILES)
    # create_classifier(CLASSIFIER_TYPES[0], 0.9)
    workbook = xlsxwriter.Workbook('ClassifierStatistics2.xlsx')
    worksheet = workbook.add_worksheet()
    text_format = workbook.add_format({'text_wrap': True})
    row = 0
    col = 0
    total_test_error = 0
    worksheet.set_column(0, 30, 20)
    mlp_dims = [[100, 100, 100], [100, 100], [100, 100, 20], [100, 20, 20], [20, 20, 20], [20, 100, 20],
                [30, 30, 30, 30],
                [100, 30, 100, 30], [30, 100, 30, 100], [40, 50, 40, 50], [20, 40, 20, 40, 20],
                [20, 40, 20, 40, 20, 40],
                [30, 50, 30, 50, 30, 50], [30, 100, 30, 100, 30], [30, 100, 30, 100, 2], [4], [5, 5]]
    num_dims = 7

    max_acc = 0
    best_layer_dim = ""

    worksheet.write(2, 0, "MLP Dimensions", text_format)
    worksheet.write(2, 1, "Average Accuracy")
    count = 0
    row = 3

    for mlp in mlp_dims:
        for i in range(3):
            print("set " + str(count) + ": attempt:" + str(i))
            total_test_error += create_classifier(CLASSIFIER_TYPES[2], 0.8, hidden_layer_sizes=mlp,
                                                  activation='relu',
                                                  max_iter=5000, alpha=1e-4,
                                                  solver='adam', verbose=0, tol=1e-6,
                                                  learning_rate_init=0.001, shuffle=True)
        total_test_error /= 3
        worksheet.write(row, col, str(mlp), text_format)
        worksheet.write(row, col + 1, round(total_test_error, 4))
        if total_test_error > max_acc:
            max_acc = total_test_error
            best_layer_dim = str(mlp)
        row += 1
        total_test_error = 0
        count += 1

    worksheet.write(0, 0, "Best MLP Dimension " + str(best_layer_dim), text_format)
    worksheet.write(0, 1, round(max_acc, 4))

    # close the book to save it
    workbook.close()
    print("max accuracy: " + str(max_acc) + " layer dimensions: " + str(best_layer_dim))
