import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

def read_object_info_file(file_path):
    """
    Read the object information from the provided txt file.
    """
    image_objects = {}

    for i in range(len(file_path)):

        with open(file_path[i], 'r') as f:

            for line in f:
                parts = line.strip().split(' ')
                if len(parts) < 2:
                    continue

                image_path = parts[0]
                objects = []

                for part in parts[1:]:
                    if len(part.split(',')) == 6:  # Ensure we have all 6 values
                        try:
                            x1, y1, x2, y2, class_id, depth = map(int, part.split(','))
                            objects.append({
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'class_id': class_id,
                                'depth': depth
                            })
                        except ValueError:
                            # Skip invalid entries
                            continue

                image_objects[image_path] = objects

    return image_objects


def get_image_height(image_path):
    """
    Get the height of the image.
    For this example, we'll use a fixed height of 375 pixels.
    In a real application, you would extract the actual height from the image file.
    """
    # You can replace this with actual image height extraction if needed
    return 375


def analyze_objects(image_objects):
    """
    Analyze objects to get distances from bottom (both for object bottom and center)
    and depths.
    """
    bottom_distances = []  # Distance from object bottom to image bottom
    center_distances = []  # Distance from object center to image bottom
    depths = []

    for image_path, objects in image_objects.items():
        image_height = get_image_height(image_path)

        for obj in objects:
            # Calculate distance from bottom (y2 is the bottom of bounding box)
            bottom_distance = image_height - obj['y2']

            # Calculate object center y-coordinate
            center_y = (obj['y1'] + obj['y2']) / 2

            # Calculate distance from center to image bottom
            center_distance = image_height - center_y

            bottom_distances.append(bottom_distance)
            center_distances.append(center_distance)
            depths.append(obj['depth'])

    return bottom_distances, center_distances, depths


def create_scatter_plots(bottom_distances, center_distances, depths):
    """
    Create two scatter plots:
    1. Object bottom distance vs depth
    2. Object center distance vs depth
    """
    # Plot 1: Bottom distance vs depth
    plt.figure(figsize=(12, 8))
    plt.scatter(bottom_distances, depths, alpha=0.7, s=50, c='blue', edgecolors='k')

    # # Add LOWESS smoothing curve
    # if len(center_distances) > 5:  # Need sufficient points for LOWESS
    #     # Sort data for smooth curve
    #     sorted_data = sorted(zip(center_distances, depths))
    #     sorted_x, sorted_y = zip(*sorted_data)
    #
    #     # Perform LOWESS smoothing (frac=0.3 controls the smoothness)
    #     smoothed = lowess(sorted_y, sorted_x, frac=0.3)
    #
    #     # Plot the smoothed curve
    #     plt.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=4)
    #
    #     # Calculate Spearman rank correlation (works better for non-linear relationships)
    #     rho, p_value = spearmanr(center_distances, depths)
    #     plt.annotate(f'Spearman ρ: {rho:.2f} (p={p_value:.3f})',
    #                  xy=(0.05, 0.95), xycoords='axes fraction',
    #                  fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Add curved trend line (quadratic polynomial fit)
    if len(bottom_distances) > 2:  # Need at least 3 points for quadratic fit
        # 使用二次多项式进行拟合 (degree=2)
        z = np.polyfit(bottom_distances, depths, 2)
        p = np.poly1d(z)
        x_range = np.linspace(min(bottom_distances), max(bottom_distances), 100)
        plt.plot(x_range, p(x_range), "r-", linewidth=4)  # 粗线宽为4的实线

        # Add R-squared value instead of correlation (more appropriate for non-linear fit)
        y_pred = p(bottom_distances)
        ss_tot = np.sum((depths - np.mean(depths)) ** 2)
        ss_res = np.sum((depths - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        plt.annotate(f'R²: {r_squared:.2f}',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.title('Relationship Between Object Bottom keypoint and Depth', fontsize=16)
    plt.xlabel('Distance from Object Bottom keypoint to Image Bottom (pixels)', fontsize=14)
    plt.ylabel('Front-back extent', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('bottom_distance_vs_depth.png', dpi=300)
    print("Plot saved as 'bottom_distance_vs_depth.png'")
    plt.show()

    # Plot 2: Center distance vs depth
    plt.figure(figsize=(12, 8))
    plt.scatter(center_distances, depths, alpha=0.7, s=50, c='green', edgecolors='k')

    # Add curved trend line (quadratic polynomial fit)
    if len(bottom_distances) > 2:  # Need at least 3 points for quadratic fit
        # 使用二次多项式进行拟合 (degree=2)
        z = np.polyfit(bottom_distances, depths, 2)
        p = np.poly1d(z)
        x_range = np.linspace(min(bottom_distances), max(bottom_distances), 100)
        plt.plot(x_range, p(x_range), "r-", linewidth=4)  # 粗线宽为4的实线

        # Add R-squared value instead of correlation (more appropriate for non-linear fit)
        y_pred = p(bottom_distances)
        ss_tot = np.sum((depths - np.mean(depths)) ** 2)
        ss_res = np.sum((depths - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        plt.annotate(f'R²: {r_squared:.2f}',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.title('Relationship Between Object Center Distance and Depth', fontsize=16)
    plt.xlabel('Distance from Object Center to Image Bottom (pixels)', fontsize=14)
    plt.ylabel('Front-back extent', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('center_distance_vs_depth.png', dpi=300)
    print("Plot saved as 'center_distance_vs_depth.png'")
    plt.show()

    return


def main():
    # Set the file path directly
    file_path = ["2012_train.txt", "2012_testt.txt", "2012_val.txt"]

    # Read object information
    print("Reading object information...")
    image_objects = read_object_info_file(file_path)

    # Check if any objects were found
    total_objects = sum(len(objects) for objects in image_objects.values())
    if total_objects == 0:
        print("No valid objects found in the file.")
        return

    print(f"Found {len(image_objects)} images with a total of {total_objects} objects.")

    # Analyze objects
    print("Analyzing objects...")
    bottom_distances, center_distances, depths = analyze_objects(image_objects)

    # Print some statistics
    print(f"\nStatistics for bottom distances:")
    print(f"Total objects analyzed: {len(bottom_distances)}")
    print(f"Average distance from bottom: {sum(bottom_distances)/len(bottom_distances):.2f} pixels")
    print(f"Min distance: {min(bottom_distances)}, Max distance: {max(bottom_distances)}")

    print(f"\nStatistics for center distances:")
    print(f"Average distance from center to bottom: {sum(center_distances)/len(center_distances):.2f} pixels")
    print(f"Min distance: {min(center_distances)}, Max distance: {max(center_distances)}")

    print(f"\nDepth statistics:")
    print(f"Average depth: {sum(depths)/len(depths):.2f}")
    print(f"Min depth: {min(depths)}, Max depth: {max(depths)}")

    # Create scatter plots
    print("\nCreating scatter plots...")
    create_scatter_plots(bottom_distances, center_distances, depths)

    print("Analysis complete!")


if __name__ == "__main__":
    main()