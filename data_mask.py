import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#DATASET
def load_data():
    # load MNIST dataset
    transform = transforms.ToTensor()
    training_data = datasets.MNIST(
        root='data', 
        train=True, 
        download=True, 
        transform=transform
        )
    # load test dataset
    test_data = datasets.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
        )

    train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    return train_loader, test_loader


#MASK

def generate_mask(image, num_pixels):
    # indices for non-zero pixels
    non_zero_indices = torch.nonzero(image.view(-1), as_tuple=True)[0]
    
    if non_zero_indices.numel() < num_pixels:
        mask_indices = non_zero_indices
    else:
        # random num_pixels indices from non-zero pixels
        mask_indices = non_zero_indices[torch.randperm(non_zero_indices.size(0))[:num_pixels]]
    
    # create mask
    mask = torch.zeros_like(image.view(-1))
    
    # selected indices to 1
    mask[mask_indices] = 1
    
    # reshape mask to original image shape
    mask = mask.view_as(image)

    # apply mask to image
    masked_image = image * mask

    return masked_image

# #TEST

# # Get some images from the test dataset
# images = test_data.data[:5].float() / 255  # Normalize the images to [0, 1]

# # Apply the generate_mask function to each image
# for i, image in enumerate(images):
#     masked_image = generate_mask(image, num_pixels=6)

#     # Plot the original and masked images
#     plt.figure(figsize=(8, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image.cpu().numpy(), cmap='gray')
#     plt.title('Original Image')
#     plt.subplot(1, 2, 2)
#     plt.imshow(masked_image.cpu().numpy(), cmap='gray')
#     plt.title('Masked Image')
#     plt.show()