import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get the current directory and image path
current_working_dir = os.path.dirname(__file__)
imgs_path = os.path.join(current_working_dir, 'images')

# Make output directory
output_path = os.path.join(current_working_dir, 'output')
os.makedirs(output_path, exist_ok=True)

# Image file names to process
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']

# Process each image
for img in images:
    # Get the base name of the image
    base_img_name = os.path.splitext(img)[0]

    # Read the image in color and grayscale
    img_color = cv2.imread(os.path.join(imgs_path, img), cv2.IMREAD_COLOR)
    img_gray = cv2.imread(os.path.join(imgs_path, img), cv2.IMREAD_GRAYSCALE)

    # Calculate each color channer energy and total energy
    blue_channel, green_channel, red_channel = cv2.split(img_color)
    blue_energy = np.sum(blue_channel)
    green_energy = np.sum(green_channel)
    red_energy = np.sum(red_channel)
    total_energy = blue_energy + green_energy + red_energy
    
    # Calculate each color's relative energy ratio
    blue_energy_ratio = float(blue_energy / total_energy)
    green_energy_ratio = float(green_energy / total_energy)
    red_energy_ratio = float(red_energy / total_energy) 

    # Plot the color energy ratios in histogram
    energy_ratios = [blue_energy_ratio, green_energy_ratio, red_energy_ratio]
    plt.bar(['Blue', 'Green', 'Red'], energy_ratios, color=['blue', 'green', 'red'])
    plt.title(f'Color Histogram - {base_img_name}')
    plt.xlabel('Color Channel')
    plt.ylabel('Energy Ratio')

    # Add text labels on top of each bar
    for i, ratio in enumerate(energy_ratios):
        plt.text(i, ratio + 0.001, f'{ratio:.3f}', ha='center', va='bottom')

    plt.savefig(os.path.join(output_path, f'{base_img_name}_color_energy_histogram.png'))
    plt.close()

    # Use matplotlib.pyplot to get and plot gray histogram
    plt.hist(img_gray.ravel(), bins=256, range=(0, 256), color='black')
    plt.title(f'Gray Histogram - {base_img_name}')
    plt.ylabel('Pixel Number')
    plt.xlabel('Pixel Value')
    plt.savefig(os.path.join(output_path, f'{base_img_name}_gray_histogram.png'))
    plt.close()

    # Calculate gradient histogram using Sobel operator
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Plot gradient histogram using matplotlib.pyplot
    plt.hist(grad_magnitude.ravel(), bins=361, range=(0, 360), color='black')
    plt.title(f'Gradient Histogram - {base_img_name}')
    plt.ylabel('Pixel Number')
    plt.xlabel('Gradient Magnitude')
    plt.savefig(os.path.join(output_path, f'{base_img_name}_gradient_histogram.png'))
    plt.close()