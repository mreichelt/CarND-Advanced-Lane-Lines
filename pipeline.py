import glob
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

show_image_interval = 3


def bgr2rgb(image):
    """Converts a BGR image (used in OpenCV) to RGB (for matplotlib)"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr2red(image):
    return image[:, :, 2]


def bgr2green(image):
    return image[:, :, 1]


def bgr2blue(image):
    return image[:, :, 0]


def bgr2hue(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    return hls[:, :, 0]


def bgr2lightness(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    return hls[:, :, 1]


def bgr2saturation(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    return hls[:, :, 2]


def bgr2gray(image):
    """Converts a BGR image to gray"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_image(image, interval=show_image_interval):
    if len(image.shape) == 3 and image.shape[2] == 3:
        plt.imshow(bgr2rgb(image))
    else:
        plt.imshow(image, cmap='gray')
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
        plt.savefig('temp/before_after.png')
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


def sobels(gray, sobel_kernel=15, thresh_x=(20, 100), thresh_y=(20, 100), thresh_mag=(30, 100), thresh_dir=(0.7, 1.3)):
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
    saturation = bgr2saturation(undistorted)
    gradx, grady, binary_magnitude, binary_direction = sobels(saturation)
    combined = sobels_combine(gradx, grady, binary_magnitude, binary_direction)
    return cv2.bitwise_and(image, image, mask=cv2.convertScaleAbs(combined))


def run_pipeline(video_file, duration=None):
    """Runs pipeline on a video and writes it to temp folder"""
    print('processing video file {}'.format(video_file))
    clip = VideoFileClip(video_file)
    if duration is not None:
        clip = clip.set_duration(duration)
    processed = clip.fl_image(pipeline)
    processed.write_videofile('temp/' + video_file, audio=False)


def main():
    plt.ion()

    images = glob.glob('test_images/test5.jpg')

    for image in images:
        image = cv2.imread(image)

        # show_gray_image(bgr2hue(undistorted))
        # show_gray_image(bgr2lightness(undistorted))
        show_image(pipeline(image))
        show_image(bgr2gray(pipeline(image)))
        show_image(bgr2saturation(image))

        # gradx, grady, binary_magnitude, binary_direction = sobels(undistorted)
        # show_cv2_image(undistorted)
        # show_gray_image(gradx)
        # show_gray_image(grady)
        # show_gray_image(binary_magnitude)
        # show_gray_image(binary_direction)
        # combined = sobels_combine(gradx, grady, binary_magnitude, binary_direction)

        # show_gray_image(pipeline(sample))

        # video_files = ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']
        # video_files = ['project_video.mp4']
        # for video_file in video_files:
        #     run_pipeline(video_file, duration=5)


main()
