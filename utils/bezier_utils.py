import numpy as np
import cv2
from PIL import Image
import random
from scipy.optimize import minimize


def draw_contours(image, contours):
    """Draw contours on a blank image with random colors."""
    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for contour in contours:
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.drawContours(output, [contour], -1, color, thickness=2)
    return output


def contour_to_bezier(contour):
    """Convert an OpenCV contour to a cubic Bézier curve with 4 control points."""
    points = contour.reshape(-1, 2)
    P0 = points[0]
    P3 = points[-1]

    def bezier_point(t, control_points):
        P0, P1, P2, P3 = control_points
        return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3

    def objective_function(params):
        P1 = np.array([params[0], params[1]])
        P2 = np.array([params[2], params[3]])
        control_points = [P0, P1, P2, P3]
        t_values = np.linspace(0, 1, len(points))
        bezier_points = np.array([bezier_point(t, control_points) for t in t_values])
        distances = np.sum((points - bezier_points) ** 2)
        return distances

    initial_P1 = P0 + (P3 - P0) / 3
    initial_P2 = P0 + 2 * (P3 - P0) / 3
    initial_guess = np.concatenate([initial_P1, initial_P2])
    result = minimize(objective_function, initial_guess, method="Nelder-Mead")

    P1 = np.array([result.x[0], result.x[1]])
    P2 = np.array([result.x[2], result.x[3]])
    return np.array([P0, P1, P2, P3])


def visualize_result(img, control_points, num_points=100):
    """Draw a Bézier curve defined by control points."""
    t_values = np.linspace(0, 1, num_points)
    curve_points = []
    for t in t_values:
        x = (
            (1 - t) ** 3 * control_points[0][0]
            + 3 * (1 - t) ** 2 * t * control_points[1][0]
            + 3 * (1 - t) * t**2 * control_points[2][0]
            + t**3 * control_points[3][0]
        )
        y = (
            (1 - t) ** 3 * control_points[0][1]
            + 3 * (1 - t) ** 2 * t * control_points[1][1]
            + 3 * (1 - t) * t**2 * control_points[2][1]
            + t**3 * control_points[3][1]
        )
        curve_points.append([int(x), int(y)])

    curve_points = np.array(curve_points, dtype=np.int32)
    for i in range(len(curve_points) - 1):
        cv2.line(img, tuple(curve_points[i]), tuple(curve_points[i + 1]), (0, 0, 0), 2)
    return img


def get_sketch(image, total_curves=10, drop_line_prob=0.0, pad=False):
    """
    Convert an image to a sketch made of Bézier curves.

    Args:
        image_path: Path to the input image
        total_curves: Total number of Bézier curves to use (default: 10)

    Returns:
        numpy.ndarray: Image with Bézier curves sketch
    """
    # Load and preprocess image
    image = np.array(image)

    # Pad image to square
    height, width, _ = image.shape
    if pad:
        max_side = max(height, width) + 20
        pad_h = (max_side - height) // 2
        pad_w = (max_side - width) // 2

        image = np.pad(
            image,
            ((pad_h, max_side - height - pad_h), (pad_w, max_side - width - pad_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

    # Convert to binary
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.where(gray_image < 255, 255, 0).astype(np.uint8)

    # Clean up binary image
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=4)

    # Get contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate curve allocation
    contour_lengths = [cv2.arcLength(contour, closed=True) for contour in contours]
    total_length = sum(contour_lengths)
    curve_allocation = [round((length / total_length) * total_curves) for length in contour_lengths]

    # Adjust allocation to match total_curves
    curve_allocation = np.clip(curve_allocation, 1, total_curves)
    while sum(curve_allocation) > total_curves:
        curve_allocation[np.argmax(curve_allocation)] -= 1
    while sum(curve_allocation) < total_curves:
        curve_allocation[np.argmin(curve_allocation)] += 1

    # Fit Bézier curves
    fitted_curves = []
    for contour, n_curves in zip(contours, curve_allocation):
        segment_length = len(contour) // n_curves
        if segment_length == 0:
            continue
        segments = [contour[i : i + segment_length] for i in range(0, len(contour), segment_length)]

        for segment in segments:
            control_points = contour_to_bezier(segment)
            fitted_curves.append(control_points)

    # Create final image
    curves_image = np.ones_like(image, dtype=np.uint8) * 255
    for curve in fitted_curves:
        if random.random() < drop_line_prob:
            continue
        curves_image = visualize_result(curves_image, curve)
    curves_image = Image.fromarray(curves_image)
    return curves_image
