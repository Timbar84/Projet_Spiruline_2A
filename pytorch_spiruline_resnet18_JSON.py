# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:21:49 2021

@author: Antoine
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy

import json
with open('parametres_ResNet18.json') as f:
    data = json.load(f)
    
## Hyper-parametres
nb_epoch = data['nb_epoch']
learning_rate = data['learning_rate']
step_size= data['step_size']
gamma= data['gamma']
batch_size = data['batch_size'] 
# Plus il est grand, il ca stabilise l'opti mais la ralentit
# Perceptron 3 multicouche classique a la fin du resnet


mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}
data_dir = 'data/spiruline_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
n_total_steps = len(dataloaders['train'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (images, labels) in enumerate(dataloaders[phase]):
                images = images.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            writer.add_scalar('accuracy_1', epoch_acc, epoch)
            writer.add_scalar('loss_1', epoch_loss, epoch)
                        
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


#### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names)) # Redimensionnement de la derniere couche
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

## Tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")

load_model = input("Chargez des paramètres déjà existants ? [0/1] ")
if load_model==1:
    model.load_state_dict(torch.load("./saves/model_1.pth"))
    model.eval()

# Entrainement
model, b_acc = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=nb_epoch)

# Sauvegarde des parametres et hyperparametres
def avoir_temps():
    return time.ctime().replace(' ','_').replace(':', '_')
def hyperparams(temps):
    ch = 'Entrainement du '+temps+'\n'
    ch+= 'nb_epoch = %d \n' % nb_epoch
    ch+= 'learning_rate = %.3f \n' % learning_rate
    ch+= 'step_size = %d \n' % step_size
    ch+= 'gamma = %.2f \n' % gamma
    ch+= 'batch_size = %d \n' % batch_size
    ch+= '~~'*10+'\n'
    return ch
b_acc = int(b_acc*1000)/1000
temps = avoir_temps()
torch.save(model.state_dict(), "./saves/model_{}_{}.pth".format(temps, b_acc))
file = open('./saves/hyperparams.txt','a')
ch = hyperparams(temps)
file.write(ch)
file.close()

writer.close()