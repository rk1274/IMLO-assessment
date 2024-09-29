import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer

from torch.optim.lr_scheduler import ReduceLROnPlateau # Scheduler for learning rate

import os # for file loading

batch_size = 32

mean = [0.4330, 0.3819, 0.2964]
std = [0.2545, 0.2044, 0.2163]

data_transforms = {
'training1' : transforms.Compose([transforms.Resize((112,112)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)]),

'training2' : transforms.Compose([transforms.Resize((112,112)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(30),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)]),

'training3' : transforms.Compose([transforms.Resize((112,112)),
                                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                                transforms.RandomPerspective(distortion_scale=0.4, p=0.6),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)]),

'testing' : transforms.Compose([transforms.Resize((112,112)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
}

train_set1 = torchvision.datasets.Flowers102(root='./data', split="train", download=True, transform=data_transforms['training1'])
train_set2 = torchvision.datasets.Flowers102(root='./data', split="train", download=True, transform=data_transforms['training2'])
train_set3 = torchvision.datasets.Flowers102(root='./data', split="train", download=True, transform=data_transforms['training3'])


big_train_set = torch.utils.data.ConcatDataset([train_set1,train_set2])
train_set = torch.utils.data.ConcatDataset([big_train_set,train_set3])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

test_set = torchvision.datasets.Flowers102(root='./data', split="test", download=True, transform=data_transforms['testing'])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

val_set = torchvision.datasets.Flowers102(root='./data', split="val", download=True, transform=data_transforms['testing'])
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)     

# Define the network
class Net(nn.Module):
	
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, 5)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(128, 256, 5) 
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 512, 5) 
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(512, 1024, 5) 
        self.bn4 = nn.BatchNorm2d(1024)

        self.fl = nn.Flatten()

        self.fc1 = nn.Linear(1024*3*3, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 102)
        self.dropout = nn.Dropout(0.75) 

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.fl(x)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)

        return x

# train_new_model creates a new network and returns a trained model.
def train_new_model():
    net = Net(l1=400, l2=350)

    optimizer, criterion, scheduler = optimLossSched(net, 0.001, 0.01)

    # train the model with epoch number, model
    net = train(2, net, optimizer, criterion, scheduler)
    print('Finished Training') 

    return net

# optimLossSched takes the model, learning rate and weight decay as parameters
# to then set and return the optimizer, loss function and scheduler.
def optimLossSched(net, lr, wd):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=8)

    return optimizer, criterion, scheduler

# train will train the model provided using a given optimizer, criterion
# and scheduler for a given number of epochs. It returns the trained model.
def train(epochs, model, optimizer, criterion, scheduler):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = run_over_training_set(model, device, optimizer, criterion)

        valid_loss = run_over_validation_set(model, device, criterion)

        display_epoch(model, epoch, optimizer, running_loss, valid_loss)

        scheduler.step(running_loss)

    return model

def run_over_training_set(model, device, optimizer, criterion):
    running_loss = 0.0

    model.train()

    for _, (images, labels) in enumerate(train_loader, 0):
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss

def run_over_validation_set(model, device, criterion):
    valid_loss = 0.0

    model.eval()

    for _, (images, labels) in enumerate(val_loader, 0):
        images = images.to(device)
        labels = labels.to(device)

        # calulate validation loss
        outputs = model(images)
        vloss = criterion(outputs, labels)
        valid_loss = vloss.item()*images.size(0)

    return valid_loss

# display_epoch displays information about each epoch
def display_epoch(model, epoch, optimizer, running_loss, valid_loss):
    valAccuracy = round(test(val_loader, model),3)

    # Display model performance after each Epoch
    print(f'Epoch {epoch+1}  -   Lr: {get_lr(optimizer)}'+
        f'  -   Training Loss: {round(running_loss / len(train_loader),3)}'+
        f'  -   Validation Loss: {round(valid_loss / len(val_loader),3)}'+
        f'  -   Val Accuracy: {valAccuracy}%')


# test will test a given dataset, data, on the given model and returns
# the accuracy of the model on the images.
def test(data, model):
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    with torch.no_grad():
        for value in data:
            images, labels = value
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return (100 * correct / total)



# get_lr returns the current learning rate from a given optimizer.
# Used in testing.
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# load_existing_model takes a path to the model that should be loaded
# and returns the model from that path.
def load_exisiting_model(PATH):
    s_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(s_dir, PATH)
    model = torch.load(file_path)

    print("Model successfully loaded")

    return model

# save_model stores the given model into the given path
def save_model(model, PATH):
    s_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(s_dir, PATH)

    torch.save(model, file_path)


if __name__ == "__main__":
    net = load_exisiting_model("best_model.pt")

    #net = train_new_model()

    print("Accuracy on test data: ",test(test_loader, net),"%")


