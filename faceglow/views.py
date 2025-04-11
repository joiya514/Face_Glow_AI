from django.http import JsonResponse
from django.shortcuts import render
from django.http import HttpResponse
import io
import base64


import sys
import os
import cv2
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
sys.path.append(os.path.abspath(os.path.join(__file__, '../../Model/GFPGAN-master/GFPGAN-master')))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../Model/GFPGAN-master/GFPGAN-master')))
from inference_gfpgan import restore_face # type: ignore



def index(request):
    return render(request, "faceglow/index.html")















def detect_face(input_image, padding_factor=0.2):
    """
    Detects and crops the face from an image, with additional padding to include more of the head and ears.

    Args:
        input_image (PIL.Image.Image or np.ndarray): Input image in PIL format or NumPy array.
        padding_factor (float): Factor to expand the bounding box for more area around the face (default is 0.2 for 20% padding).

    Returns:
        dict: Dictionary containing the cropped face, bounding box, and rotation angle (if any).
    """
    # If the input image is a NumPy array, convert it to a PIL image
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    # Ensure the input is in RGB format
    image = input_image.convert('RGB')

    # Initialize MTCNN detector
    mtcnn = MTCNN(keep_all=False, post_process=False)  # Single face detection

    # Detect face and get bounding box
    bounding_box, _ = mtcnn.detect(image)

    if bounding_box is not None:
        # Extract coordinates of the bounding box
        x1, y1, x2, y2 = map(int, bounding_box[0])

        # Calculate padding to expand the bounding box
        width = x2 - x1
        height = y2 - y1
        padding_x = int(padding_factor * width)  # Padding on the left and right
        padding_y = int(padding_factor * height)  # Padding on top and bottom

        # Update bounding box coordinates to include the padding
        x1 = max(0, x1 - padding_x)  # Ensure the box doesn't go out of bounds
        y1 = max(0, y1 - padding_y)
        x2 = min(image.width, x2 + padding_x)
        y2 = min(image.height, y2 + padding_y)

        # Crop the face with the updated bounding box
        cropped_face = image.crop((x1, y1, x2, y2))

        # Create a dictionary to return all relevant details
        face_info = {
            'cropped_face': cropped_face,
            'bounding_box': [x1, y1, x2, y2],  # Bounding box coordinates
            'width': x2 - x1,  # Width of the bounding box
            'height': y2 - y1,  # Height of the bounding box
            # Optionally, you can add more details such as the rotation angle if needed
            'angle': 0  # Set to 0 or calculate if needed (e.g., using landmarks or rotation matrix)
        }
        return face_info
    else:
        return {'cropped_face': image, 'bounding_box': None, 'angle': 0}





def denoise_face(cropped_face, h=10):
    """
    Denoise the cropped face image using Non-Local Means Denoising.

    Args:
        cropped_face (PIL.Image.Image or np.ndarray): Cropped face image (in RGB or BGR).
        h (float): Parameter controlling the denoising strength. Higher values remove more noise but may blur the image.

    Returns:
        denoised_face (PIL.Image.Image): Denoised face image.
    """
    # If the input image is a PIL image, convert it to a NumPy array
    if isinstance(cropped_face, Image.Image):
        cropped_face = np.array(cropped_face)

    # If the image is in RGB format, convert it to BGR (for OpenCV compatibility)
    if cropped_face.shape[-1] == 3:  # Check if it's an RGB image
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
    
    # Apply Non-Local Means Denoising (works well for faces)
    denoised_face = cv2.fastNlMeansDenoisingColored(cropped_face, None, h, h, 7, 21)

    # Convert back to RGB if necessary
    denoised_face = cv2.cvtColor(denoised_face, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array back to PIL Image
    denoised_face = Image.fromarray(denoised_face)

    return denoised_face



def integrate_restored_face(original_image, restored_face, bounding_box, angle=0):
    """
    Integrate the restored face back into the original image.
    """
    x1, y1, x2, y2 = bounding_box
    box_width = x2 - x1
    box_height = y2 - y1

    # Resize restored face to match bounding box dimensions
    restored_face_resized = cv2.resize(restored_face, (box_width, box_height))

    # Rotate the resized face back to match original orientation
    center = (restored_face_resized.shape[1] // 2, restored_face_resized.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
    restored_face_corrected = cv2.warpAffine(
        restored_face_resized,
        rotation_matrix,
        (restored_face_resized.shape[1], restored_face_resized.shape[0]),
    )

    # Create a copy of the original image to integrate the restored face
    integrated_image = original_image.copy()

    # Ensure the restored face fits perfectly into the bounding box
    
    # Convert to NumPy array
    integrated_image_np = np.array(integrated_image)
    # Do the assignment
    integrated_image_np[y1:y2, x1:x2] = restored_face_corrected
    # Convert back to PIL Image
    integrated_image = Image.fromarray(integrated_image_np)

    return integrated_image





def sharpen_preserve_color(image, alpha=1.5, beta=-0.5, gamma=0):
    """
    Sharpen the image while preserving its color by applying sharpening only to the luminance channel.

    Args:
        image (np.ndarray): Input image in BGR format.
        alpha (float): Weight of the original image (default is 1.5 for sharpening).
        beta (float): Weight of the blurred image (default is -0.5 for sharpening).
        gamma (float): Scalar added to each sum (default is 0 for no additional brightness).

    Returns:
        np.ndarray: Sharpened image in BGR format with color preserved.
    """
    # Convert the image to Lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # Split into L (lightness), a, and b channels
    l, a, b = cv2.split(lab)

    # Apply Gaussian blur to the L-channel
    blurred_l = cv2.GaussianBlur(l, (5, 5), 0)

    # Sharpen the L-channel
    sharpened_l = cv2.addWeighted(l, alpha, blurred_l, beta, gamma)

    # Clip the values to ensure they're valid (0-255)
    sharpened_l = np.clip(sharpened_l, 0, 255).astype(np.uint8)

    # Merge the modified L-channel back with the original a and b channels
    sharpened_lab = cv2.merge((sharpened_l, a, b))

    # Convert the Lab image back to BGR color space
    sharpened_bgr = cv2.cvtColor(sharpened_lab, cv2.COLOR_Lab2BGR)

    return sharpened_bgr




import cv2
import numpy as np

def enhance_color_and_brightness(image, brightness_factor=1.15, contrast_clip_limit=1.2, tile_grid_size=(8, 8), color_boost_factor=1.1):
    """
    Enhances the color and brightness of an image while maintaining a natural look.

    Args:
        image (numpy.ndarray): Input image (BGR format).
        brightness_factor (float): Factor to adjust brightness. >1 increases brightness, <1 decreases brightness.
        contrast_clip_limit (float): Clip limit for CLAHE (higher values increase contrast).
        tile_grid_size (tuple): Tile grid size for CLAHE.
        color_boost_factor (float): Factor to boost the color intensity (greater than 1 to make colors more vivid).

    Returns:
        numpy.ndarray: Image with enhanced color and brightness.
    """
    # Convert BGR image to LAB color space for contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L (lightness) channel to improve contrast
    clahe = cv2.createCLAHE(clipLimit=contrast_clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    
    # Merge CLAHE-enhanced L-channel with a and b channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    
    # Convert back to BGR for color adjustment
    enhanced_color = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # Convert the enhanced image to HSV for brightness adjustment
    hsv = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2HSV)
    
    # Split the HSV channels
    h, s, v = cv2.split(hsv)
    
    # Adjust brightness by scaling the V (value) channel, ensuring it stays within valid range
    v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)
    
    # Slightly boost the saturation (S channel) to make the colors more vivid
    s = np.clip(s * color_boost_factor, 0, 255).astype(np.uint8)
    
    # Merge the adjusted channels back
    hsv_adjusted = cv2.merge((h, s, v))
    
    # Convert back to BGR to get the final enhanced image
    enhanced_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    return enhanced_image




def show_images(actual_image, enhanced_image):

    # Ensure the images are of the same size
    if actual_image.shape != enhanced_image.shape:
        print("Resizing images to the same dimensions.")
        enhanced_image = cv2.resize(enhanced_image, (actual_image.shape[1], actual_image.shape[0]))

    # Function to update the display based on slider position
    def update_slider(x):
        # Get the slider position
        split = cv2.getTrackbarPos('Split Position', 'Image Comparison')
        # Create a combined image
        combined = np.hstack((actual_image[:, :split], enhanced_image[:, split:]))
        
        # Resize combined image to fit window while maintaining aspect ratio
        resized_combined = cv2.resize(combined, (resize_width, resize_height))
        # Show the resized combined image
        cv2.imshow('Image Comparison', resized_combined)

    # Create a window
    cv2.namedWindow('Image Comparison')

    # Calculate the desired window size based on image dimensions, but limit the size
    max_width = 800  # Max width of the display window
    max_height = 600  # Max height of the display window

    # Calculate the aspect ratio of the actual image
    aspect_ratio = actual_image.shape[1] / actual_image.shape[0]
    
    # Adjust the window size to fit within the max dimensions, preserving aspect ratio
    if actual_image.shape[1] > actual_image.shape[0]:
        resize_width = min(actual_image.shape[1], max_width)
        resize_height = int(resize_width / aspect_ratio)
    else:
        resize_height = min(actual_image.shape[0], max_height)
        resize_width = int(resize_height * aspect_ratio)

    # Ensure the window size doesn't exceed the max dimensions
    resize_width = min(resize_width, max_width)
    resize_height = min(resize_height, max_height)

    # Resize the window to the calculated dimensions
    cv2.resizeWindow('Image Comparison', resize_width, resize_height)

    # Minimize the window initially
    cv2.setWindowProperty('Image Comparison', cv2.WND_PROP_FULLSCREEN, 0)

    # Add a slider (trackbar)
    cv2.createTrackbar('Split Position', 'Image Comparison', actual_image.shape[1] // 2, actual_image.shape[1], update_slider)

    # Show the initial state
    update_slider(actual_image.shape[1] // 2)

    # Wait until the user presses a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_valid_image():
    """
    Prompt the user to enter an image path, validate the input, and return the loaded image.
    The user can enter -1 to exit the loop.
    
    Returns:
        image (np.ndarray): Loaded image in BGR format.
        image_path (str): Path to the valid image.
        Returns None if the user exits.
    """
    while True:
        # Prompt user to enter the image path
        image_path = input("Enter the path to the image (or -1 to quit): ").strip()
        
        # Check if the user wants to exit
        if image_path == "-1":
            print("Exiting...")
            return None
        
        # Check if the file exists
        if not os.path.isfile(image_path):
            print("Error: File not found. Please try again.")
            continue
        
        # Try loading the image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print("Error: The file is not a valid image. Please try again.")
            continue
        
        image_name = image_path.split("\\")[-1]
        # If valid, return the image and path
        print("Image loaded successfully!")
        return image, image_name



def enhance_image(request):
    file = request.FILES.get("image")
    image = Image.open(file)
    # Unpack the dictionary returned by detect_face
    face_info = detect_face(image)
    face = face_info['cropped_face']
    box = face_info['bounding_box']
    angle = face_info['angle']

    if box is None:
        # Handle no face detected case
        print("No face detected")
        integrated_image = image
    else:
        # Denoise the cropped face
        denoised = denoise_face(face)

        # Integrate the denoised face into the original image
        integrated_image = integrate_restored_face(image, np.array(denoised), box, angle)

    # Now send the integrated image to the restore_face function
    integrated_image = np.array(integrated_image)
    # Convert RGB to BGR for OpenCV processing
    if integrated_image.shape[2] == 3:  # Ensure it's a 3-channel image
        integrated_image = cv2.cvtColor(integrated_image, cv2.COLOR_RGB2BGR)

    cropped_face, restored_face, restored_img = restore_face(integrated_image)
    
    sharpened_image = sharpen_preserve_color(restored_img)
    enhanced_image = enhance_color_and_brightness(sharpened_image)
    
    cv2.imwrite('E:\\Sixth Semester\\Enhanced.jpeg', enhanced_image)

    enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    # Convert numpy array to image
    image = Image.fromarray(enhanced_image_rgb.astype('uint8'))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)

    # Convert to base64 encoding
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return JsonResponse({'image': 'data:image/png;base64,' + image_base64})

def save_image(image, name, path):

    ext = name.split(".")[-1]
    name = name.split(".")[0]
    print(name)
    print(ext)
    name = name + "_enhanced." + ext
    print(name)
    path = os.path.join(path, name)
    cv2.imwrite(path, image)
    print(f"Image saved to '{path}'")