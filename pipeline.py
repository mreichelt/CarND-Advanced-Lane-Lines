import numpy as np
import cv2
import pickle
import PyQt5
import matplotlib.pyplot as plt

show_image_interval = 3


def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_cv2_image(image, interval=show_image_interval):
    plt.imshow(bgr2rgb(image))
    plt.pause(interval)
    plt.close()


def show_before_after(before, after, before_title='', after_title='', interval=show_image_interval):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(bgr2rgb(before))
    ax1.set_title(before_title, fontsize=30)
    ax2.imshow(bgr2rgb(after))
    ax2.set_title(after_title, fontsize=30)
    plt.show()
    plt.pause(interval)
    plt.close()


def load_first_image_of_video(video_file):
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    if ret:
        cap.release()
        return frame
    else:
        exit('unable to read video ' + video_file)


def load_camera_calibration(file='wide_dist_pickle.p'):
    return pickle.load(open(file, 'rb'))


def undistort(image, calibration):
    return cv2.undistort(image, calibration['mtx'], calibration['dist'], None, calibration['mtx'])


def main():
    plt.ion()
    calibration = load_camera_calibration()
    sample = load_first_image_of_video('project_video.mp4')
    undistorted = undistort(sample, calibration)
    show_before_after(sample, undistorted)


main()
