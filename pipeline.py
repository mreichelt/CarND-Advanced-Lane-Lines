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
    Minv = cv2.getPerspectiveTransform(get_perspective_transform_dst(), get_perspective_transform_src())
    binary_warped = cv2.warpPerspective(mask, M, image_size(image), flags=cv2.INTER_LINEAR)

    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low)
                          & (nonzeroy < win_y_high)
                          & (nonzerox >= win_xleft_low)
                          & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low)
                           & (nonzeroy < win_y_high)
                           & (nonzerox >= win_xright_low)
                           & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.pause(1.5)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, image_size(undistorted))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    # Now our radius of curvature is in meters
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius = (left_curverad + right_curverad) / 2
    cv2.putText(result, 'radius: {:5.0f}m'.format(radius), (10, 50),
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result


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

    # images = glob.glob('test_images/*.jpg')
    #
    # for image in images:
    #     image = cv2.imread(image)
    #     show_image(pipeline(image))

    print('encoding video')

    # video_files = ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']
    video_files = ['project_video.mp4']
    for video_file in video_files:
        run_pipeline(video_file)


main()
