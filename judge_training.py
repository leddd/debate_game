import torch
from torch import nn, optim
from input_data import train_loader, generate_mask
from judge import Judge
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#DATASET

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

# Generate mask
def generate_mask(image, num_pixels):
    # Get the indices of the non-zero pixels
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


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")








# MODEL

class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
    
model = Judge().to(device)





# TRAINING

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (images, labels) in enumerate(dataloader):
        # Generate and apply mask
        mask = generate_mask(images, 6)  # 6 nonzero pixels
        masked_images = images * mask

        pred = model(masked_images, mask)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            mask = generate_mask(images, 6)
            images = images * mask
            outputs = model(images, mask)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 64
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
print("Done!")




# Save the model state
torch.save(model.state_dict(), 'Judge.pth')
print("Saved Judge Model State to Judge.pth")