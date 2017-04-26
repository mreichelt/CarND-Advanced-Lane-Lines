import numpy as np
import cv2
import glob
import pickle
import PyQt5
import matplotlib.pyplot as plt

# run interactively
plt.ion()

# number of chessboard corners (rows, columns), other parameters
nx = 9
ny = 6
dir = 'camera_cal'
pattern = 'calibration*.jpg'
calibration_file = 'wide_dist_pickle.p'

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob(dir + '/' + pattern)

# Step through the list and search for chessboard corners
for file in images:
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # write_name = 'corners_found'+str(idx)+'.jpg'
        # cv2.imwrite(write_name, img)
        plt.imshow(img)
        plt.pause(1)
    else:
        print('chessboard corners not found for ' + file)

print('calibrating camera…')

# now let's calibrate the camera
image_size = (1280, 720)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
# save it to a file
pickle.dump({'mtx': mtx, 'dist': dist}, open(calibration_file, 'wb'))

print('showing undistorted images…')

# check visually how undistorted images look like
for file in images:
    img = cv2.imread(file)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()
    plt.pause(1)
