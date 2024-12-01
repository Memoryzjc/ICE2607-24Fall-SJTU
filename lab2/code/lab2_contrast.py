import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def gradients(img, op='sobel'):
    # Choose different gradient operators to get the gradients
    if op == 'sobel':
        # Use Sobel operator to get the gradients
        grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    elif op == 'scharr':
        grad_x = cv2.Scharr(img, cv2.CV_16S, 1, 0)
        grad_y = cv2.Scharr(img, cv2.CV_16S, 0, 1)
    elif op == 'laplacian':
        grad_x = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        grad_y = cv2.Laplacian(img, cv2.CV_16S, ksize=3)

    # Convert the gradients to absolute values
    grad_x_abs = cv2.convertScaleAbs(grad_x)
    grad_y_abs = cv2.convertScaleAbs(grad_y)

    # Convert the gradients to float32
    grad_x_abs = np.float32(grad_x_abs)
    grad_y_abs = np.float32(grad_y_abs)

    # Calculate the magnitude of the gradients and the angle
    magnitude = np.sqrt(grad_x_abs**2 + grad_y_abs**2)
    angle = np.arctan2(grad_y, grad_x)

    return magnitude, angle

def interpolate(M1, M2, w):
    # Calculate the interpolated value
    return M1 * w + M2 * (1 - w)
    # if (M1 > M2):
    #     return w * M1 + (1 - w) * M2
    # else:
    #     return w * M1 + (1 - w) * M2

def non_max_suppression_interpolated(magnitude, angle):
    # Non-maximum suppression with interpolation
    rows, cols = magnitude.shape
    suppressed_interpolated = np.zeros((rows, cols), dtype=np.float32)

    # Preprocess the angle
    angle = np.rad2deg(angle) % 180

    # Interpolated
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            # Check the angle
            if angle[i, j] == 0:
                if magnitude[i, j] > magnitude[i, j - 1] and magnitude[i, j] > magnitude[i, j + 1]:
                    suppressed_interpolated[i, j] = magnitude[i, j]
            elif angle[i, j] == 135:
                if magnitude[i, j] > magnitude[i - 1, j + 1] and magnitude[i, j] > magnitude[i + 1, j - 1]:
                    suppressed_interpolated[i, j] = magnitude[i, j]
            elif angle[i, j] == 90:
                if magnitude[i, j] > magnitude[i - 1, j] and magnitude[i, j] > magnitude[i + 1, j]:
                    suppressed_interpolated[i, j] = magnitude[i, j]
            elif angle[i, j] == 45:
                if magnitude[i, j] > magnitude[i - 1, j - 1] and magnitude[i, j] > magnitude[i + 1, j + 1]:
                    suppressed_interpolated[i, j] = magnitude[i, j]
            elif 0 < angle[i, j] < 45:
                w = np.tan(np.deg2rad(angle[i, j]))
                dtmp1_mag = interpolate(magnitude[i, j - 1], magnitude[i - 1, j - 1], w)
                dtmp2_mag = interpolate(magnitude[i, j + 1], magnitude[i + 1, j + 1], w)
                if magnitude[i, j] > dtmp1_mag and magnitude[i, j] > dtmp2_mag:
                    suppressed_interpolated[i, j] = magnitude[i, j]
            elif 45 < angle[i, j] < 90:
                w = 1 / np.tan(np.deg2rad(angle[i, j]))
                dtmp1_mag = interpolate(magnitude[i - 1, j], magnitude[i - 1, j - 1], w)
                dtmp2_mag = interpolate(magnitude[i + 1, j], magnitude[i + 1, j + 1], w)
                if magnitude[i, j] > dtmp1_mag and magnitude[i, j] > dtmp2_mag:
                    suppressed_interpolated[i, j] = magnitude[i, j]
            elif 90 < angle[i, j] < 135:
                w = -1 / np.tan(np.deg2rad(angle[i, j]))
                dtmp1_mag = interpolate(magnitude[i - 1, j], magnitude[i - 1, j + 1], w)
                dtmp2_mag = interpolate(magnitude[i + 1, j], magnitude[i + 1, j - 1], w)
                if magnitude[i, j] > dtmp1_mag and magnitude[i, j] > dtmp2_mag:
                    suppressed_interpolated[i, j] = magnitude[i, j]
            elif 135 < angle[i, j] < 180:
                w = -np.tan(np.deg2rad(angle[i, j]))
                dtmp1_mag = interpolate(magnitude[i, j - 1], magnitude[i - 1, j + 1], w)
                dtmp2_mag = interpolate(magnitude[i, j + 1], magnitude[i + 1, j - 1], w)
                if magnitude[i, j] > dtmp1_mag and magnitude[i, j] > dtmp2_mag:
                    suppressed_interpolated[i, j] = magnitude[i, j]

    return suppressed_interpolated


def non_max_suppression(magnitude, angle):
    # Non-maximum suppression
    M, N = magnitude.shape
    suppressed = np.zeros((M, N), dtype=np.float32)
    angle = np.rad2deg(angle) % 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i - 1, j]
                r = magnitude[i + 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
    return suppressed

def double_threshold(img, low_threshold, high_threshold):
    # Double threshold
    strong = 255
    weak = 75

    # Get the strong, weak, and zeros pixels' indices
    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img >= low_threshold) & (img < high_threshold))
    zeros_i, zeros_j = np.where(img < low_threshold)

    # Set the pixels to strong, weak, and zeros
    img[strong_i, strong_j] = strong
    img[weak_i, weak_j] = weak
    img[zeros_i, zeros_j] = 0

    return img, weak, strong

def edge_tracking(img, weak, strong=255):
    # Edge tracking
    M, N = img.shape
    for i in range(3, M - 3):
        for j in range(3, N - 3):
            if img[i, j] == weak:
                if strong in [img[i + a, j + b] for a in [-3, -2, -1, 0, 1, 2, 3] for b in [-3, -2, -1, 0, 1, 2, 3]]:
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def canny_edge_detection(img, low_threshold, high_threshold, op='sobel'):
    # Canny edge detection algorithm
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), .5)

    # Get the gradients and angles
    magnitude, angles = gradients(img_blur, op=op)

    # Non-maximum suppression
    suppressed = non_max_suppression(magnitude, angles)

    # Double threshold
    thresholded, weak, strong = double_threshold(suppressed, low_threshold, high_threshold)

    # Edge tracking
    edges = edge_tracking(thresholded, weak, strong)

    return edges

def canny_edge_detection_interpolated(img, low_threshold, high_threshold):
    # Canny edge detection algorithm
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0.5)

    # Get the gradients and angles
    magnitude, angles = gradients(img_blur)

    # Non-maximum suppression
    suppressed = non_max_suppression_interpolated(magnitude, angles)
    plt.figure()
    # Double threshold
    thresholded, weak, strong = double_threshold(suppressed, low_threshold, high_threshold)

    # Edge tracking
    edges = edge_tracking(thresholded, weak, strong)

    return edges


if __name__ == '__main__':
    # Get the path of the current working directory
    current_working_dir = os.path.dirname(__file__)
    imgs_path = os.path.join(current_working_dir, 'img')

    # Make output directory
    output_path = os.path.join(current_working_dir, 'output')
    os.makedirs(output_path, exist_ok=True)

    # Iterate over the images
    imgs = ('1.jpg', '2.jpg', '3.jpg')

    # Set the high threshold and the ratio
    h_th = 150
    ratio = 0.45
    l_th = int(ratio * h_th)

    for index, i in enumerate(imgs):
        img = cv2.imread(os.path.join(imgs_path, i))
        edges_low = canny_edge_detection(img, low_threshold=l_th, high_threshold=h_th)
        edges_interpolated = canny_edge_detection_interpolated(img, low_threshold=l_th, high_threshold=h_th)
        edge = cv2.Canny(img, l_th, h_th)

        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(edges_low, cmap='gray')
        plt.title('Canny Edge Detection-My Implementation')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(edges_interpolated, cmap='gray')
        plt.title('Canny Edge Detection-Interpolated')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(edge, cmap='gray')
        plt.title('Canny Edge Detection-OpenCV')
        plt.axis('off')

        plt.suptitle(f'Canny Edge Detection Comparison-High Threshold: {h_th}, Ratio: {ratio}-image:{index+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'canny_edge_detection_image{index+1}_HighThreshold_{h_th}_Ratio_{ratio}.png'))
        plt.close()

        sobel_edge = canny_edge_detection(img, low_threshold=l_th, high_threshold=h_th, op='sobel')
        scharr_edge = canny_edge_detection(img, low_threshold=l_th, high_threshold=h_th, op='scharr')
        laplacian_edge = canny_edge_detection(img, low_threshold=l_th, high_threshold=h_th, op='laplacian')
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(sobel_edge, cmap='gray')
        plt.title('Canny Edge Detection-Sobel')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(scharr_edge, cmap='gray')
        plt.title('Canny Edge Detection-Scharr')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(laplacian_edge, cmap='gray')
        plt.title('Canny Edge Detection-Laplacian')
        plt.axis('off')

        plt.suptitle(f'Canny Edge Detection Comparison-Graident Operator(High Threshold: {h_th}, Ratio: {ratio}-image:{index+1})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'canny_edge_detection_image{index+1}_HighThreshold_{h_th}_Ratio_{ratio}_GradientOperator.png'))