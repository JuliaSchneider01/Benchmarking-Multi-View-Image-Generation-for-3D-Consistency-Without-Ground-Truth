import cv2
import numpy as np
import random
import os
from matplotlib import pyplot as plt
from PIL import Image

def hue_shift(image_rgb: np.ndarray) -> np.ndarray:
    """
    Applies a global hue shift to an RGB image.

    Parameters:
        image_rgb (np.ndarray): Input RGB image as a NumPy array in [0, 255].

    Returns:
        np.ndarray: Hue-shifted RGB image as a NumPy array in [0, 255].
    """
    # Convert RGB to HSV (OpenCV uses BGR, so we convert carefully)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Hue shift sampled uniformly from [0, 1)
    delta = random.uniform(0, 1)

    # OpenCV stores H in [0, 179], corresponding to [0, 360) degrees
    delta_cv = delta * 180

    # Apply global hue shift with modulo wrapping
    image_hsv[..., 0] = (image_hsv[..., 0] + delta_cv) % 180

    # Convert back to RGB
    image_hsv = np.clip(image_hsv, 0, 255).astype(np.uint8)
    image_rgb_shifted = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    return image_rgb_shifted



def save_hue_shifted_image(input_path: str):
    """
    Loads an image, applies hue shift, and saves it with "_hue" in the filename.
    """
    # Load image as RGB
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Apply hue shift
    shifted_rgb = hue_shift(image_rgb)

    # Convert back to BGR for saving with OpenCV
    shifted_bgr = cv2.cvtColor(shifted_rgb, cv2.COLOR_RGB2BGR)

    # Create output path with "_hue"
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_hue{ext}"

    # Save image
    cv2.imwrite(output_path, shifted_bgr)
    print(f"Saved hue-shifted image to: {output_path}")


def plot_images(original: np.ndarray, shifted: np.ndarray):
    """
    Plots original and hue-shifted images side by side.
    Both images expected as RGB NumPy arrays.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Hue Shifted")
    plt.imshow(shifted)
    plt.axis('off')

    plt.show()


def main():
    # Load image using PIL and convert to NumPy RGB
    img_path = "FML_hard/photoconsistent-nvs/samples/000c3ab189999a83/samples/00000000/images/0005.png"
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)

    # Apply hue shift
    shifted = hue_shift(image_np)

    plot_images(original=image_np, shifted=shifted)
    save_hue_shifted_image(img_path)

if __name__ == "__main__":
    main()
