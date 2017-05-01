import glob
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

show_image_interval = 1.5
image_counter = 1


def next_image_filename():
    global image_counter
    file = 'temp/{}.jpg'.format(image_counter)
    image_counter += 1
    return file


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

    plt.savefig(next_image_filename())
    plt.pause(interval)
    plt.close()


def show_before_after(before, after, before_title='', after_title='', interval=show_image_interval):
    """Shows two images (before/after comparison) for a short interval and closes"""
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(bgr2rgb(before))
    ax1.set_title(before_title, fontsize=30)
    ax2.imshow(bgr2rgb(after))
    ax2.set_title(after_title, fontsize=30)

    plt.savefig(next_image_filename())
    plt.show()
    plt.pause(interval)
    plt.close()


def get_perspective_transform_src(image_width=1280, image_height=720):
    top = 454
    top_left = 588
    top_right = 694
    bottom_right = 1117
    bottom_left = 194

    vertices = np.float32([[
        (bottom_left, image_height),  # bottom left
        (top_left, top),  # top left
        (top_right, top),  # top right
        (bottom_right, image_height)  # bottom right
    ]])
    return vertices


def get_perspective_transform_dst(image_width=1280, image_height=720):
    left = 300
    right = image_width - left

    vertices = np.float32([[
        (left, image_height),  # bottom left
        (left, 0),  # top left
        (right, 0),  # top right
        (right, image_height)  # bottom right
    ]])
    return vertices


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


def abs_sobel_threshold(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mask = np.zeros_like(scaled)
    mask[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return mask


def magnitude_treshold(gray, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    scaled = np.uint8(255 * magnitude / np.max(magnitude))
    mask = np.zeros_like(scaled)
    mask[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return mask


def direction_treshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    mask = np.zeros_like(direction)
    mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return mask


def image_size(image):
    return (image.shape[1], image.shape[0])


def mask1_and_mask2(mask1, mask2):
    merged = np.zeros_like(mask1)
    merged[(mask1 == 1) & (mask2 == 1)] = 1
    return merged


def mask1_or_mask2(mask1, mask2):
    merged = np.zeros_like(mask1)
    merged[(mask1 == 1) | (mask2 == 1)] = 1
    return merged


def pipeline(image):
    undistorted = undistort(image)

    red = bgr2red(undistorted)
    magnitude_red = magnitude_treshold(red, thresh=(20, 100))
    direction_red = direction_treshold(red, thresh=(0.7, 1.3))
    red_mask = mask1_and_mask2(magnitude_red, direction_red)

    saturation = bgr2saturation(undistorted)
    magnitude_saturation = magnitude_treshold(saturation, thresh=(40, 100))
    direction_saturation = direction_treshold(saturation, thresh=(0.7, 1.3))
    saturation_mask = mask1_and_mask2(magnitude_saturation, direction_saturation)

    mask = mask1_or_mask2(red_mask, saturation_mask)

    # cv2.polylines(undistorted, get_perspective_transform_src().astype(int), True, (0, 0, 255), thickness=1)
    M = cv2.getPerspectiveTransform(get_perspective_transform_src(), get_perspective_transform_dst())
    warped = cv2.warpPerspective(mask, M, image_size(image), flags=cv2.INTER_LINEAR)

    return warped


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
        show_image(pipeline(image))

        # video_files = ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']
        # video_files = ['project_video.mp4']
        # for video_file in video_files:
        #     run_pipeline(video_file)


main()
