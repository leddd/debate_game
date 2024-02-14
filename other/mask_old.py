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

train_loader, test_loader = load_data()

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

    return mask, masked_image

# # #TEST

# # get a batch of data
# images, labels = next(iter(train_loader))

# # for each of the first 5 images in the batch
# for i in range(5):
#     # select the image
#     image = images[i]

#     # generate a mask for the image
#     mask, masked_image = generate_mask(image, num_pixels=6)

#     # plot the original, mask and masked images
#     fig, ax = plt.subplots(1, 3)

#     # original image
#     ax[0].imshow(image.squeeze().numpy(), cmap='gray')
#     ax[0].set_title('Original Image')

#     # mask
#     ax[1].imshow(mask.squeeze().numpy(), cmap='gray')
#     ax[1].set_title('Mask')

#     # masked image
#     ax[2].imshow(masked_image.squeeze().numpy(), cmap='gray')
#     ax[2].set_title('Masked Image')

#     plt.show()