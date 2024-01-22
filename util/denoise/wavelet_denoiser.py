import pywt
import torch
import numpy as np

from torchvision import transforms
from PIL import Image

def wavelet_denoise(data, wavelet, level):
    # Decompose to get the wavelet coefficients
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Calculate a threshold
    sigma = np.median(np.abs(coeffs[-1]))
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # Apply thresholding
    new_coeffs = map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs)

    # Reconstruct the signal
    return pywt.waverec(list(new_coeffs), wavelet)

# Example usage

# Path to your image
image_path = '/home/jakobtroidl/Desktop/NVP/data/hemibrain-large-mip-1/train/15242_15703_17155.png'

# Read the image
image = Image.open(image_path)

# Define a transformation to convert the image to a tensor
transform = transforms.ToTensor()

# Apply the transformation to the image
image_tensor = transform(image)

wavelet = 'db1'  # Daubechies wavelet
level = 2  # Level of decomposition

# Denoise
denoised_data = wavelet_denoise(image_tensor, wavelet, level)

# Convert to PyTorch tensor if needed
denoised_data_tensor = torch.tensor(denoised_data)


# safe denoised data tensor as png image
denoised_image = transforms.ToPILImage()(denoised_data_tensor)
denoised_image.save('15242_15703_17155_denoised.png')
