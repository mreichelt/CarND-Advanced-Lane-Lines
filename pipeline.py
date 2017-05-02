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
    height = image_height - top

    vertices = np.float32([[
        (bottom_left, image_height),  # bottom left
        (top_left, top),  # top left
        (top_right, top),  # top right
        (bottom_right, image_height)  # bottom right
    ]])
    return vertices, height


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


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def get_poly(self):
        return self.current_fit[0] * self.ally ** 2 + self.current_fit[1] * self.ally + self.current_fit[2]

    def update(self, current_fit, line_base_pos, radius_of_curvature, img_size):
        self.current_fit = current_fit
        self.line_base_pos = line_base_pos
        self.radius_of_curvature = radius_of_curvature
        self.ally = np.linspace(0, img_size[1] - 1, img_size[1])
        self.allx = self.get_poly()


class LineHistory:
    def __init__(self, n=10):
        self.n = n
        self.left_lines = []
        self.right_lines = []

    def append(self, left: Line, right: Line):
        self.left_lines.append(left)
        self.right_lines.append(right)

    def take_last(self):
        return self.left_lines[-self.n:], self.right_lines[-self.n:]

    def averages(self, img_size):
        left_lines, right_lines = self.take_last()
        return self.line_average(left_lines, img_size), self.line_average(right_lines, img_size)

    def line_average(self, lines, img_size):
        line = Line()
        current_fit = np.average([line.current_fit for line in lines], axis=0)
        line_base_pos = np.average([line.line_base_pos for line in lines])
        radius_of_curvature = np.average([line.radius_of_curvature for line in lines])
        line.update(current_fit, line_base_pos, radius_of_curvature, img_size)
        return line


def detect_lines(binary_warped, n_windows=6, margin=100, minpix_recenter=50):
    left = Line()
    right = Line()

    height = binary_warped.shape[0]
    width = binary_warped.shape[1]
    histogram = np.sum(binary_warped[int(height / 2):, :], axis=0)
    histogram_width = histogram.shape[0]
    window_height = np.int(height / n_windows)

    # Create an output image to draw on and  visualize the result
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram_width / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_indices = []
    right_lane_indices = []

    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_indices = ((nonzeroy >= win_y_low)
                             & (nonzeroy < win_y_high)
                             & (nonzerox >= win_xleft_low)
                             & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_indices = ((nonzeroy >= win_y_low)
                              & (nonzeroy < win_y_high)
                              & (nonzerox >= win_xright_low)
                              & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_indices) > minpix_recenter:
            leftx_current = np.int(np.mean(nonzerox[good_left_indices]))
        if len(good_right_indices) > minpix_recenter:
            rightx_current = np.int(np.mean(nonzerox[good_right_indices]))

    # Concatenate the arrays of indices
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_indices]
    lefty = nonzeroy[left_lane_indices]
    rightx = nonzerox[right_lane_indices]
    righty = nonzeroy[right_lane_indices]

    # Fit a second order polynomial to each
    left.current_fit = np.polyfit(lefty, leftx, 2)
    right.current_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ally = np.linspace(0, height - 1, height)
    left.ally = right.ally = ally
    left.allx = left.get_poly()
    right.allx = right.get_poly()

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    y_eval = np.max(ally)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    left.line_base_pos = (width / 2 - leftx_base) * xm_per_pix
    right.line_base_pos = (rightx_base - width / 2) * xm_per_pix

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ally * ym_per_pix, left.allx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ally * ym_per_pix, right.allx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left.radius_of_curvature = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) \
                               / np.absolute(2 * left_fit_cr[0])
    right.radius_of_curvature = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) \
                                / np.absolute(2 * right_fit_cr[0])

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.pause(1.5)

    return left, right


def get_line_mask(undistorted):
    red = bgr2red(undistorted)
    magnitude_red = magnitude_treshold(red, thresh=(20, 100))
    direction_red = direction_treshold(red, thresh=(0.7, 1.3))
    red_mask = mask1_and_mask2(magnitude_red, direction_red)

    saturation = bgr2saturation(undistorted)
    magnitude_saturation = magnitude_treshold(saturation, thresh=(40, 100))
    direction_saturation = direction_treshold(saturation, thresh=(0.7, 1.3))
    saturation_mask = mask1_and_mask2(magnitude_saturation, direction_saturation)

    mask = mask1_or_mask2(red_mask, saturation_mask)

    return mask


def pipeline(image, line_history: LineHistory):
    undistorted = undistort(image)

    mask = get_line_mask(undistorted)

    src, src_height = get_perspective_transform_src()
    dst = get_perspective_transform_dst()
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = image_size(image)
    binary_warped = cv2.warpPerspective(mask, M, img_size, flags=cv2.INTER_LINEAR)

    # detect the lines in warped mode
    left, right = detect_lines(binary_warped)

    # now, lets get a smooth value over the history of lines
    line_history.append(left, right)
    left_average, right_average = line_history.averages(img_size)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_average.allx, left_average.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_average.allx, right_average.ally])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, image_size(undistorted))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    # draw radius + distance from center on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius = (left_average.radius_of_curvature + right_average.radius_of_curvature) / 2
    cv2.putText(result, 'radius: {:5.2f}km'.format(radius / 1000), (10, 30),
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    center_distance = (right_average.line_base_pos - left_average.line_base_pos) / 2
    cv2.putText(result, 'distance from center: {:5.2f}m'.format(center_distance), (10, 60),
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result


def run_pipeline(video_file, duration=None, end=False):
    """Runs pipeline on a video and writes it to temp folder"""
    print('processing video file {}'.format(video_file))
    clip = VideoFileClip(video_file)

    if duration is not None:
        if end:
            clip = clip.subclip(clip.duration - duration)
        else:
            clip = clip.subclip(0, duration)

    line_history = LineHistory()
    processed = clip.fl(lambda gf, t: pipeline(gf(t), line_history), [])
    processed.write_videofile('temp/' + video_file, audio=False)


def main():
    plt.ion()

    do_images = True
    do_videos = False

    if do_images:
        images = glob.glob('test_images/*.jpg')

        for image in images:
            image = cv2.imread(image)
            show_image(pipeline(image, LineHistory()))

    if do_videos:
        # video_files = ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']
        video_files = ['project_video.mp4']
        for video_file in video_files:
            run_pipeline(video_file)
            # run_pipeline(video_file, duration=14, end=True)


main()
