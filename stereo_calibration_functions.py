import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from google.colab import files
from google.colab.patches import cv2_imshow
import pickle


def calibrate_single_camera(images_folder, chessboard_size=(10, 14), square_size=1.0):
    """
    Calibrate a single camera using chessboard images.

    Args:
        images_folder: Path to folder containing calibration images
        chessboard_size: Internal corners of chessboard (columns, rows)
        square_size: Size of chessboard squares in real world units

    Returns:
        camera_matrix, distortion_coefficients, rvecs, tvecs, rms_error
    """
    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points - 3D points in real world space
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Load images
    images = glob.glob(images_folder + '/*.jpg') + glob.glob(images_folder + '/*.png')

    if not images:
        raise ValueError(f"No images found in {images_folder}")

    img_shape = None
    successful_detections = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (width, height)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)

            # Refine corner positions to sub-pixel accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            successful_detections += 1

            # Optionally visualize detected corners
            # cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            # cv2.imshow('Corners', img)
            # cv2.waitKey(100)

    # cv2.destroyAllWindows()

    if successful_detections < 10:
        print(f"Warning: Only {successful_detections} successful detections. Recommend at least 10.")

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print(f"Camera calibration RMS error: {ret:.4f}")
    print(f"Successful detections: {successful_detections}/{len(images)}")

    return mtx, dist, rvecs, tvecs, ret


def stereo_calibrate(left_images_folder, right_images_folder, left_camera_matrix, left_dist_coeffs,
                    right_camera_matrix, right_dist_coeffs, chessboard_size=(10, 14), square_size=15.0):
    """
    Perform stereo calibration using synchronized image pairs.

    Returns:
        Stereo calibration parameters including R, T, E, F matrices
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    # Load synchronized image pairs
    left_images = sorted(glob.glob(left_images_folder + '/*.jpg') + glob.glob(left_images_folder + '/*.png'))
    right_images = sorted(glob.glob(right_images_folder + '/*.jpg') + glob.glob(right_images_folder + '/*.png'))

    if len(left_images) != len(right_images):
        raise ValueError("Number of left and right images must be equal")

    img_shape = None
    successful_pairs = 0

    for left_fname, right_fname in zip(left_images, right_images):
        left_img = cv2.imread(left_fname)
        right_img = cv2.imread(right_fname)

        if left_img is None or right_img is None:
            continue

        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        img_shape = left_gray.shape[::-1]

        # Find chessboard corners in both images
        ret_left, corners_left = cv2.findChessboardCorners(left_gray, chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(right_gray, chessboard_size, None)

        if ret_left and ret_right:
            objpoints.append(objp)

            # Refine corners
            corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)

            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            successful_pairs += 1

    print(f"Successful stereo pairs: {successful_pairs}")

    # Stereo calibration
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        left_camera_matrix, left_dist_coeffs,
        right_camera_matrix, right_dist_coeffs,
        img_shape, criteria=stereocalib_criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f"Stereo calibration RMS error: {ret:.4f}")

    return ret, mtx1, dist1, mtx2, dist2, R, T, E, F, img_shape

def stereo_rectify(mtx1, dist1, mtx2, dist2, img_shape, R, T):
    """
    Compute rectification transforms for stereo image pair.

    Returns:
        Rectification and projection matrices, and rectification maps
    """
    # Stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx1, dist1, mtx2, dist2, img_shape, R, T,
        alpha=0.9  # 0=full crop, 1=no crop
    )

    # Compute rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        mtx1, dist1, R1, P1, img_shape, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        mtx2, dist2, R2, P2, img_shape, cv2.CV_16SC2
    )

    return R1, R2, P1, P2, Q, roi1, roi2, left_map1, left_map2, right_map1, right_map2

def rectify_image_pair(left_img, right_img, left_map1, left_map2, right_map1, right_map2):
    """
    Apply rectification to stereo image pair.
    """
    left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LANCZOS4)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LANCZOS4)

    return left_rectified, right_rectified

def create_stereo_matcher(method='SGBM'):
    """
    Create stereo matcher with optimized parameters.

    Args:
        method: 'BM' for Block Matching or 'SGBM' for Semi-Global Block Matching

    Returns:
        Configured stereo matcher
    """
    if method == 'BM':
        # Block Matching parameters
        stereo = cv2.StereoBM_create()
        stereo.setNumDisparities(64)  # Must be divisible by 16
        stereo.setBlockSize(15)  # Odd number, typically 5-21
        stereo.setPreFilterCap(31)
        stereo.setPreFilterSize(9)
        stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        stereo.setTextureThreshold(10)
        stereo.setUniquenessRatio(10)
        stereo.setSpeckleRange(32)
        stereo.setSpeckleWindowSize(100)
        stereo.setDisp12MaxDiff(1)
        stereo.setMinDisparity(0)

    elif method == 'SGBM':
        # Semi-Global Block Matching parameters (generally better quality)
        window_size = 5
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,  # Must be divisible by 16
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    return stereo

def compute_disparity(left_img, right_img, matcher):
    """
    Compute disparity map from rectified stereo pair.

    Returns:
        Disparity map normalized to 0-255 range
    """
    # Convert to grayscale if needed
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img

    # Compute disparity
    disparity = matcher.compute(left_gray, right_gray)

    # Convert from fixed-point representation
    disparity = disparity.astype(np.float32) / 16.0

    # Normalize for visualization
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return disparity, disparity_normalized


def disparity_to_depth(disparity, Q_matrix, baseline=None, focal_length=None):
    """
    Convert disparity map to depth map.

    Args:
        disparity: Disparity map
        Q_matrix: 4x4 reprojection matrix from stereo rectification
        baseline: Distance between cameras (optional, extracted from Q if not provided)
        focal_length: Focal length (optional, extracted from Q if not provided)

    Returns:
        Depth map in same units as baseline
    """
    # Method 1: Using reprojectImageTo3D (recommended)
    points_3D = cv2.reprojectImageTo3D(disparity, Q_matrix)
    depth_map = points_3D[:, :, 2]  # Z coordinate is depth

    # Method 2: Manual calculation using formula: depth = (baseline * focal_length) / disparity
    if baseline is not None and focal_length is not None:
        # Avoid division by zero
        disparity_safe = np.where(disparity > 0, disparity, 0.001)
        depth_manual = (baseline * focal_length) / disparity_safe

        # Set invalid depths to a large value
        depth_manual[disparity <= 0] = 1000

        return depth_map, depth_manual

    return depth_map

def filter_depth_map(depth_map, max_depth=10.0, min_depth=0.1):
    """
    Filter depth map to remove outliers and invalid values.
    """
    depth_filtered = depth_map.copy()

    # Remove negative depths and extreme values
    depth_filtered[depth_filtered < min_depth] = max_depth
    depth_filtered[depth_filtered > max_depth] = max_depth
    depth_filtered[np.isnan(depth_filtered)] = max_depth
    depth_filtered[np.isinf(depth_filtered)] = max_depth

    return depth_filtered

def visualize_results(left_img, right_img, disparity_norm, depth_map, save_path=None):
    """
    Create visualization of stereo vision results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Original left image
    axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Left Image')
    axes[0, 0].axis('off')

    # Original right image
    axes[0, 1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Right Image')
    axes[0, 1].axis('off')

    # Disparity map
    im1 = axes[1, 0].imshow(disparity_norm, cmap='jet')
    axes[1, 0].set_title('Disparity Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Depth map
    im2 = axes[1, 1].imshow(depth_map, cmap='plasma')
    axes[1, 1].set_title('Depth Map')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def draw_epipolar_lines(left_rectified, right_rectified, num_lines=20):
    """
    Draw epipolar lines to verify rectification quality.
    """
    height = left_rectified.shape[0]

    # Create copies for drawing
    left_with_lines = left_rectified.copy()
    right_with_lines = right_rectified.copy()

    # Draw horizontal lines
    for i in range(0, height, height // num_lines):
        cv2.line(left_with_lines, (0, i), (left_with_lines.shape[1], i), (0, 255, 0), 1)
        cv2.line(right_with_lines, (0, i), (right_with_lines.shape[1], i), (0, 255, 0), 1)

    # Display side by side
    combined = np.hstack([left_with_lines, right_with_lines])

    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Images with Epipolar Lines')
    plt.axis('off')
    plt.show()