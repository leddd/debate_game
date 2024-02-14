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

    return mask, masked_image


# get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# MODEL
class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1) #2 input planes
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(3*3*256, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

    
# initialize the model and optimizer
model = Judge().to(device)
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
batch_size = 128
total_batches = 150000  # Total number of batches to train on
criterion = nn.CrossEntropyLoss()


# load data
train_loader, test_loader = load_data()


# training loop

model.train()
total_loss = 0
batch_count = 0

for epoch in range(total_batches // len(train_loader) + 1):  # number of epochs
    for images, labels in train_loader:
        # stop training after total_batches
        if batch_count >= total_batches:
            break
        
        # generate and apply masks
        masks, masked_images = zip(*[generate_mask(image, num_pixels=6) for image in images])
        masks = torch.stack(masks).to(device)
        masked_images = torch.stack(masked_images).to(device) 
        labels = labels.to(device)

        # stack masks and masked_images along the channel dimension
        inputs = torch.cat((masks, masked_images), dim=1)

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # total_loss += loss.item()
        batch_count += 1

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    


# save the trained model
torch.save(model.state_dict(), 'model.pth')
