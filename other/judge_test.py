import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from judge_training import load_data, generate_mask, Judge

# get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# load the trained model
model = Judge()
model.load_state_dict(torch.load('judge.pth'))
model.to(device)
model.eval()

# load data
_, test_loader = load_data()

# initialize the test loss and the correct predictions count
test_loss = 0
correct = 0
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in test_loader:
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

        # update test loss
        test_loss += loss.item()

        # get the index of the max log-probability
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

# print test loss and accuracy
test_loss /= len(test_loader.dataset)
print(f'Test Loss: {test_loss:.6f}')
print(f'Test Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')


# Test Loss: 0.008444
# Test Accuracy: 6165/10000 (62%)
