import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#DATASET

# Load MNIST dataset
transform = transforms.ToTensor()
training_data = datasets.MNIST(
    root='data', 
    train=True, 
    download=True, 
    transform=transform
    )
# Load the test dataset
test_data = datasets.MNIST(
    root='data', 
    train=False, 
    download=True, 
    transform=transform
    )

train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

def generate_mask(image, num_pixels):
    # Indices of the non-zero pixels
    non_zero_indices = torch.nonzero(image.view(-1), as_tuple=True)[0]
    
    if non_zero_indices.numel() < num_pixels:
        mask_indices = non_zero_indices
    else:
        # Randomly select num_pixels indices from the non-zero pixels
        mask_indices = non_zero_indices[torch.randperm(non_zero_indices.size(0))[:num_pixels]]
    
    # Create a mask
    mask = torch.zeros_like(image.view(-1))
    
    # Set the selected indices to 1
    mask[mask_indices] = 1
    
    return mask.view_as(image)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

images = test_data.data[:128].float() / 255

# Mask test
for i in range(5):
    image = images[i]
    mask = generate_mask(image, 6)
    masked_image = image * mask

    # Plot original image, mask, and masked image
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().numpy(), cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask.cpu().numpy(), cmap='gray')
    plt.title('Mask')
    plt.subplot(1, 3, 3)
    plt.imshow(masked_image.cpu().numpy(), cmap='gray')
    plt.title('Masked Image')
    plt.show()