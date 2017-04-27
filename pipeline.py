import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

show_image_interval = 1


def bgr2rgb(image):
    """Converts a BGR image (used in OpenCV) to RGB (for matplotlib)"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr2gray(image):
    """Converts a BGR image to gray"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_cv2_image(image, interval=show_image_interval, cmap=None):
    """Shows an image for a short interval and closes"""
    plt.imshow(bgr2rgb(image), cmap=cmap)
    plt.pause(interval)
    plt.close()


def show_gray_image(gray, interval=show_image_interval):
    """Shows a gray image for a short interval and closes"""
    plt.imshow(gray, cmap='gray')
    plt.pause(interval)
    plt.close()


def show_before_after(before, after, before_title='', after_title='', interval=show_image_interval, write_file=True):
    """Shows two images (before/after comparison) for a short interval and closes"""
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(bgr2rgb(before))
    ax1.set_title(before_title, fontsize=30)
    ax2.imshow(bgr2rgb(after))
    ax2.set_title(after_title, fontsize=30)
    if write_file:
        plt.savefig('temp.jpg')
    plt.show()
    plt.pause(interval)
    plt.close()


def load_first_image_of_video(video_file):
    """Returns the first image of a video"""
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    if ret:
        cap.release()
        return frame
    else:
        exit('unable to read video ' + video_file)


def load_camera_calibration(file='wide_dist_pickle.p'):
    """Loads the camera calibration"""
    return pickle.load(open(file, 'rb'))


calibration_global = load_camera_calibration()


def undistort(image, calibration=calibration_global):
    """Corrects a distorted image"""
    return cv2.undistort(image, calibration['mtx'], calibration['dist'], None, calibration['mtx'])


def sobels(image, sobel_kernel=15, thresh_x=(20, 100), thresh_y=(20, 100), thresh_mag=(30, 100), thresh_dir=(0.7, 1.3)):
    gray = bgr2gray(image)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))

    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    direction = np.arctan2(abs_sobely, abs_sobelx)

    gradx = np.zeros_like(abs_sobelx)
    grady = np.zeros_like(scaled_sobely)
    binary_magnitude = np.zeros_like(scaled_magnitude)
    binary_direction = np.zeros_like(direction)

    gradx[(scaled_sobelx >= thresh_x[0]) & (scaled_sobelx < thresh_x[1])] = 1
    grady[(scaled_sobely >= thresh_y[0]) & (scaled_sobely < thresh_y[1])] = 1
    binary_magnitude[(scaled_magnitude >= thresh_mag[0]) & (scaled_magnitude < thresh_mag[1])] = 1
    binary_direction[(direction >= thresh_dir[0]) & (direction < thresh_dir[1])] = 1

    return gradx, grady, binary_magnitude, binary_direction


def sobels_combine(gradx, grady, mag_binary, dir_binary):
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def pipeline(image):
    undistorted = undistort(image)
    gradx, grady, binary_magnitude, binary_direction = sobels(undistorted)
    combined = sobels_combine(gradx, grady, binary_magnitude, binary_direction)
    return cv2.bitwise_and(image, image, mask=cv2.convertScaleAbs(combined))


def main():
    plt.ion()

    sample = load_first_image_of_video('project_video.mp4')
    # undistorted = undistort(sample)
    # gradx, grady, binary_magnitude, binary_direction = sobels(undistorted)
    # show_cv2_image(undistorted)
    # show_gray_image(gradx)
    # show_gray_image(grady)
    # show_gray_image(binary_magnitude)
    # show_gray_image(binary_direction)
    # combined = sobels_combine(gradx, grady, binary_magnitude, binary_direction)

    # show_gray_image(pipeline(sample))

    clip = VideoFileClip('project_video.mp4')
    clip = clip.set_duration(5)
    processed = clip.fl_image(pipeline)
    processed.write_videofile('temp.mp4', audio=False)


main()
