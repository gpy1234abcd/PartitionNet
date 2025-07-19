import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# 设置全局字体为 Palatino Linotype
plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['font.size'] = 12


def format_coefficient(coef, precision=6):
    """
    格式化系数，自动选择最佳显示方式（小数或科学计数法）

    Args:
        coef: 系数值
        precision: 精度位数

    Returns:
        格式化后的系数字符串
    """
    abs_coef = abs(coef)

    # 如果系数很小（小于10^-4）或很大（大于10^4），使用科学计数法
    if abs_coef < 1e-4 or abs_coef > 1e4:
        # 使用科学计数法，但显示为 ×10^n 的形式
        formatted = f"{coef:.{precision-1}e}"
        # 将 'e' 格式转换为 '×10^' 格式
        if 'e' in formatted:
            mantissa, exponent = formatted.split('e')
            exponent = int(exponent)
            return f"{mantissa}×10^{{{exponent}}}"
        return formatted

    # 对于正常范围的数值，使用高精度小数格式
    elif abs_coef < 1e-6:
        return f"{coef:.{precision+2}f}"  # 更高精度
    else:
        return f"{coef:.{precision}f}"


def format_polynomial_equation(coefficients, precision=6):
    """
    格式化多项式方程，使用高精度系数显示

    Args:
        coefficients: 多项式系数数组（从最高次项到常数项）
        precision: 系数精度

    Returns:
        格式化后的多项式方程字符串
    """
    poly_str = "y = "
    degree = len(coefficients) - 1

    for i, coef in enumerate(coefficients):
        power = degree - i

        # 跳过接近零的系数（但保留常数项）
        if abs(coef) < 1e-12 and power != 0:
            continue

        # 添加符号
        if i > 0:
            if coef >= 0:
                poly_str += " + "
            else:
                poly_str += " - "
                coef = abs(coef)
        elif coef < 0:
            poly_str += "-"
            coef = abs(coef)

        # 格式化系数
        coef_str = format_coefficient(coef, precision)

        # 添加项
        if power == 0:
            poly_str += coef_str
        elif power == 1:
            if abs(coef - 1.0) < 1e-10:  # 系数为1时
                poly_str += "x"
            else:
                poly_str += f"{coef_str}x"
        else:
            if abs(coef - 1.0) < 1e-10:  # 系数为1时
                poly_str += f"x^{{{power}}}"
            else:
                poly_str += f"{coef_str}x^{{{power}}}"

    return poly_str


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
    """
    return 375


def analyze_objects(image_objects):
    """
    Analyze objects to get distances and depths.
    """
    bottom_distances = []  # Distance from object bottom to image bottom
    center_distances = []  # Distance from object center to image bottom
    depths = []

    for image_path, objects in image_objects.items():
        image_height = get_image_height(image_path)

        for obj in objects:
            # Calculate distance from bottom
            bottom_distance = image_height - obj['y2']

            # Calculate object center y-coordinate
            center_y = (obj['y1'] + obj['y2']) / 2

            # Calculate distance from center to image bottom
            center_distance = image_height - center_y

            bottom_distances.append(bottom_distance)
            center_distances.append(center_distance)
            depths.append(obj['depth'])

    return bottom_distances, center_distances, depths


def find_best_polynomial_degree(x_data, y_data, max_degree=4):
    """
    Find the best polynomial degree using cross-validation and information criteria.
    Limited to 1-4 degrees to avoid overfitting and unnatural curves.
    """
    degrees = range(1, max_degree + 1)
    cv_scores = []
    aic_scores = []
    bic_scores = []
    r2_scores = []

    X = np.array(x_data).reshape(-1, 1)
    y = np.array(y_data)
    n = len(y)

    print(f"Testing polynomial degrees from 1 to {max_degree} (avoiding overfitting)...")
    print("-" * 90)
    print(f"{'Degree':<8} {'Type':<12} {'R²':<8} {'CV Score':<12} {'AIC':<12} {'BIC':<12} {'MSE':<12}")
    print("-" * 90)

    for degree in degrees:
        # Define polynomial type names
        poly_types = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic"}
        poly_type = poly_types.get(degree, f"Degree-{degree}")

        # Create polynomial pipeline
        poly_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])

        # Fit the model
        poly_pipeline.fit(X, y)
        y_pred = poly_pipeline.predict(X)

        # Calculate R²
        r2 = r2_score(y, y_pred)
        r2_scores.append(r2)

        # Calculate MSE
        mse = mean_squared_error(y, y_pred)

        # Cross-validation score (negative MSE)
        cv_score = cross_val_score(poly_pipeline, X, y, cv=5, scoring='neg_mean_squared_error').mean()
        cv_scores.append(-cv_score)  # Convert back to positive MSE

        # Calculate AIC and BIC
        # For linear regression: AIC = n*ln(MSE) + 2*k, BIC = n*ln(MSE) + k*ln(n)
        # where k is the number of parameters (degree + 1)
        k = degree + 1
        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + k * np.log(n)

        aic_scores.append(aic)
        bic_scores.append(bic)

        print(f"{degree:<8} {poly_type:<12} {r2:<8.4f} {-cv_score:<12.4f} {aic:<12.2f} {bic:<12.2f} {mse:<12.4f}")

    # Find best degrees according to different criteria
    best_r2_idx = np.argmax(r2_scores)
    best_cv_idx = np.argmin(cv_scores)
    best_aic_idx = np.argmin(aic_scores)
    best_bic_idx = np.argmin(bic_scores)

    print("-" * 90)
    poly_types = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic"}
    print(
        f"Best degree by R²: {degrees[best_r2_idx]} ({poly_types[degrees[best_r2_idx]]}) - R² = {r2_scores[best_r2_idx]:.4f}")
    print(
        f"Best degree by CV: {degrees[best_cv_idx]} ({poly_types[degrees[best_cv_idx]]}) - CV MSE = {cv_scores[best_cv_idx]:.4f}")
    print(
        f"Best degree by AIC: {degrees[best_aic_idx]} ({poly_types[degrees[best_aic_idx]]}) - AIC = {aic_scores[best_aic_idx]:.2f}")
    print(
        f"Best degree by BIC: {degrees[best_bic_idx]} ({poly_types[degrees[best_bic_idx]]}) - BIC = {bic_scores[best_bic_idx]:.2f}")

    # Recommend the degree that balances complexity and performance
    # Usually BIC is more conservative and helps avoid overfitting
    recommended_degree = degrees[best_bic_idx]

    print(f"\n🎯 Recommended degree: {recommended_degree} ({poly_types[recommended_degree]}) - based on BIC criterion")
    print(f"📊 This avoids overfitting while maintaining good predictive performance")

    return recommended_degree, {
        'degrees': degrees,
        'r2_scores': r2_scores,
        'cv_scores': cv_scores,
        'aic_scores': aic_scores,
        'bic_scores': bic_scores
    }


def plot_model_selection_results(results):
    """
    Plot the model selection results.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    degrees = results['degrees']

    # R² plot
    ax1.plot(degrees, results['r2_scores'], 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Polynomial Degree', fontfamily='Palatino Linotype')
    ax1.set_ylabel('R² Score', fontfamily='Palatino Linotype')
    ax1.set_title('R² vs Polynomial Degree', fontfamily='Palatino Linotype')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(degrees)

    # Cross-validation MSE plot
    ax2.plot(degrees, results['cv_scores'], 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Polynomial Degree', fontfamily='Palatino Linotype')
    ax2.set_ylabel('Cross-Validation MSE', fontfamily='Palatino Linotype')
    ax2.set_title('CV MSE vs Polynomial Degree', fontfamily='Palatino Linotype')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(degrees)

    # AIC plot
    ax3.plot(degrees, results['aic_scores'], 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Polynomial Degree', fontfamily='Palatino Linotype')
    ax3.set_ylabel('AIC', fontfamily='Palatino Linotype')
    ax3.set_title('AIC vs Polynomial Degree', fontfamily='Palatino Linotype')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(degrees)

    # BIC plot
    ax4.plot(degrees, results['bic_scores'], 'mo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Polynomial Degree', fontfamily='Palatino Linotype')
    ax4.set_ylabel('BIC', fontfamily='Palatino Linotype')
    ax4.set_title('BIC vs Polynomial Degree', fontfamily='Palatino Linotype')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(degrees)

    plt.tight_layout()
    plt.savefig('model_selection_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_optimized_scatter_plots(bottom_distances, center_distances, depths):
    """
    Create optimized scatter plots with best polynomial fits and high-precision coefficients.
    """
    # Find best polynomial degrees
    print("=" * 80)
    print("ANALYZING BOTTOM DISTANCE vs DEPTH")
    print("=" * 80)
    best_degree_bottom, results_bottom = find_best_polynomial_degree(bottom_distances, depths)

    print("\n" + "=" * 80)
    print("ANALYZING CENTER DISTANCE vs DEPTH")
    print("=" * 80)
    best_degree_center, results_center = find_best_polynomial_degree(center_distances, depths)

    # Plot model selection results
    print("\nPlotting model selection results...")
    plot_model_selection_results(results_bottom)

    # Plot 1: Bottom distance vs depth with optimal polynomial
    plt.figure(figsize=(12, 8))
    plt.scatter(bottom_distances, depths, alpha=0.7, s=50, c='blue', edgecolors='k', label='Data points')

    if len(bottom_distances) > best_degree_bottom:
        # Fit optimal polynomial
        z = np.polyfit(bottom_distances, depths, best_degree_bottom)
        p = np.poly1d(z)

        # Create smooth curve within data range (avoid extrapolation issues)
        x_min, x_max = min(bottom_distances), max(bottom_distances)
        # Add small buffer to avoid edge effects
        x_buffer = (x_max - x_min) * 0.02
        x_range = np.linspace(x_min + x_buffer, x_max - x_buffer, 200)
        y_fit = p(x_range)

        plt.plot(x_range, y_fit, "r-", linewidth=4,
                 label=f'Polynomial fit (degree {best_degree_bottom})')

        # Calculate R²
        y_pred = p(bottom_distances)
        r_squared = r2_score(depths, y_pred)

        # 使用改进的高精度格式化函数
        poly_types = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic"}
        poly_type = poly_types.get(best_degree_bottom, f"Degree-{best_degree_bottom}")

        # 使用新的格式化函数
        poly_str = format_polynomial_equation(z, precision=8)

        # 修改：移除 fontsize 参数，使用默认字体大小（12），与图例保持一致
        plt.annotate(f'Best fit: {poly_type} (Degree {best_degree_bottom})\nR² = {r_squared:.6f}\n{poly_str}',
                     xy=(0.05, 0.70), xycoords='axes fraction',
                     fontfamily='Palatino Linotype',
                     bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="navy", alpha=0.9))

        # 在控制台打印高精度系数
        print(f"\n高精度系数 (Bottom Distance vs Depth):")
        print(f"多项式阶数: {best_degree_bottom}")
        for i, coef in enumerate(z):
            power = len(z) - 1 - i
            print(f"x^{power} 系数: {coef:.12e}")

    # plt.title('Optimized Relationship: Object Bottom Distance vs Depth', fontsize=16, fontfamily='Palatino Linotype')
    plt.xlabel('Distance from Object Bottom to Image Bottom (pixels)', fontsize=14, fontfamily='Palatino Linotype')
    plt.ylabel('Depth', fontsize=14, fontfamily='Palatino Linotype')
    plt.legend(prop={'family': 'Palatino Linotype'})
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('optimized_bottom_distance_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Center distance vs depth with optimal polynomial
    plt.figure(figsize=(12, 8))
    plt.scatter(center_distances, depths, alpha=0.7, s=50, c='green', edgecolors='k', label='Data points')

    if len(center_distances) > best_degree_center:
        # 修正：使用center_distances而不是bottom_distances
        z = np.polyfit(center_distances, depths, best_degree_center)
        p = np.poly1d(z)

        # Create smooth curve within data range (avoid extrapolation issues)
        x_min, x_max = min(center_distances), max(center_distances)
        # Add small buffer to avoid edge effects
        x_buffer = (x_max - x_min) * 0.02
        x_range = np.linspace(x_min + x_buffer, x_max - x_buffer, 200)
        y_fit = p(x_range)

        plt.plot(x_range, y_fit, "r-", linewidth=4,
                 label=f'Polynomial fit (degree {best_degree_center})')

        # Calculate R²
        y_pred = p(center_distances)
        r_squared = r2_score(depths, y_pred)

        # 使用改进的高精度格式化函数
        poly_types = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic"}
        poly_type = poly_types.get(best_degree_center, f"Degree-{best_degree_center}")

        # 使用新的格式化函数
        poly_str = format_polynomial_equation(z, precision=8)

        # 修改：移除 fontsize 参数，使用默认字体大小（12），与图例保持一致
        plt.annotate(f'Best fit: {poly_type} (Degree {best_degree_center})\nR² = {r_squared:.6f}\n{poly_str}',
                     xy=(0.05, 0.70), xycoords='axes fraction',
                     fontfamily='Palatino Linotype',
                     bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="darkgreen", alpha=0.9))

        # 在控制台打印高精度系数
        print(f"\n高精度系数 (Center Distance vs Depth):")
        print(f"多项式阶数: {best_degree_center}")
        for i, coef in enumerate(z):
            power = len(z) - 1 - i
            print(f"x^{power} 系数: {coef:.12e}")

    # plt.title('Optimized Relationship: Object Center Distance vs Depth', fontsize=16, fontfamily='Palatino Linotype')
    plt.xlabel('Distance from Object Center to Image Bottom (pixels)', fontsize=14, fontfamily='Palatino Linotype')
    plt.ylabel('Depth', fontsize=14, fontfamily='Palatino Linotype')
    plt.legend(prop={'family': 'Palatino Linotype'})
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('optimized_center_distance_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.show()

    return best_degree_bottom, best_degree_center


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

    # Print statistics
    print(f"\nStatistics:")
    print(f"Total objects analyzed: {len(bottom_distances)}")
    print(f"Bottom distance - Mean: {np.mean(bottom_distances):.2f}, Std: {np.std(bottom_distances):.2f}")
    print(f"Center distance - Mean: {np.mean(center_distances):.2f}, Std: {np.std(center_distances):.2f}")
    print(f"Depth - Mean: {np.mean(depths):.2f}, Std: {np.std(depths):.2f}")

    # Create optimized scatter plots
    print("\nFinding optimal polynomial fits...")
    best_bottom_degree, best_center_degree = create_optimized_scatter_plots(
        bottom_distances, center_distances, depths)

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS:")
    print("=" * 80)
    poly_types = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic"}
    print(f"📊 Best polynomial for bottom distance: {poly_types[best_bottom_degree]} (degree {best_bottom_degree})")
    print(f"📊 Best polynomial for center distance: {poly_types[best_center_degree]} (degree {best_center_degree})")
    print(f"")
    print(f"🔍 Why we limit to 1-4 degrees:")
    print(f"   • Degree 1 (Linear): Simple straight line relationship")
    print(f"   • Degree 2 (Quadratic): Parabolic curve, most common in physics")
    print(f"   • Degree 3 (Cubic): S-shaped curves, good for complex relationships")
    print(f"   • Degree 4 (Quartic): More complex curves, but still manageable")
    print(f"   • Higher degrees often cause overfitting and unnatural curves!")
    print("✅ Analysis complete! Check the generated plots and model selection results.")


if __name__ == "__main__":
    main()