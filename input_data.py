import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST dataset
transform = transforms.ToTensor()
training_data = datasets.MNIST(
    root='data', 
    train=True, 
    download=True, 
    transform=transform
    )
test_data = datasets.MNIST(
    root='data', 
    train=False, 
    download=True, 
    transform=transform
    )

train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

def generate_mask(image, num_pixels):
    # Get indices of the non-zero pixels
    non_zero_indices = torch.nonzero(image.view(-1), as_tuple=True)[0]
    
    if non_zero_indices.numel() < num_pixels:
        mask_indices = non_zero_indices
    else:
        # Randomly select num_pixels indices from the non-zero pixels
        mask_indices = non_zero_indices[torch.randperm(non_zero_indices.size(0))[:num_pixels]]
    
    # Create a mask of zeros
    mask = torch.zeros_like(image.view(-1))
    
    # Set the selected indices to 1
    mask[mask_indices] = 1
    
    return mask.view_as(image)