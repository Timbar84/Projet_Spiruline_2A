# -*- coding: utf-8 -*-

# ------------------------------------------------------------------- #
# ---------------           Importations              --------------- #
# ------------------------------------------------------------------- #
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
from Generation_image_et_generateur import *
import json
with open('parametres_ResNet18.json') as f:
    data = json.load(f)
    
# ------------------------------------------------------------------- #
# ---------------            Paramètres               --------------- #
# ------------------------------------------------------------------- #
## Hyper-parametres récupérés d'un fichier JSON
pretrained = bool(data['pretrained'])
nb_epoch = data['nb_epoch']
learning_rate = data['learning_rate']
step_size= data['step_size']
gamma= data['gamma']
train_size = data['train_size']
val_size = data['val_size']
batch_size = data['batch_size']
nombre_classe = data['nombre_classe']

# Machine = CPU ou GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------- #
# ---------------            fonctions                --------------- #
# ------------------------------------------------------------------- #
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Iterate over epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                size = train_size
            else:
                model.eval()   # Set model to evaluate mode
                size = val_size

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for images, labels in generateur_image(size, batch_size):
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
                running_corrects += torch.sum(preds == labels)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / size
            epoch_acc = running_corrects.double() / size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            writer.add_scalar('accuracy', epoch_acc, epoch)
            writer.add_scalar('loss', epoch_loss, epoch)
                        
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

# ------------------------------------------------------------------- #
# ---------------            Programme                --------------- #
# ------------------------------------------------------------------- #

model = models.resnet18(pretrained=pretrained) # Choix du modèle
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, nombre_classe) # Redimensionnement de la derniere couche
model = model.to(device)

# Choix du critère
criterion = nn.CrossEntropyLoss() 
# Choix de l'optimiseur
optimizer = optim.SGD(model.parameters(), lr=learning_rate) 
# Choix du scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

## Tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")

# Entrainement
model, b_acc = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=nb_epoch)

writer.close()