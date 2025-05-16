import torchvision
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import PIL
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models


if __name__ == '__main__':
    
    #Cargamos las imágenes y las máscaras
    images = os.listdir('/home/22061506mercedes/ComputerVision_UNET/data_flood/Image')
    masks = os.listdir('/home/22061506mercedes/ComputerVision_UNET/data_flood/Mask')

    print(len(images), len(masks))

    image_tensor = list()
    masks_tensor = list()
    for image in images:
        #Las imágenes están en jpg, por lo que las convertimos a png
        dd = PIL.Image.open(f'/home/22061506mercedes/ComputerVision_UNET/data_flood/Image{image}')
        #Lo pasamos a tensor
        tt = torchvision.transforms.functional.pil_to_tensor(dd)
        tt = torchvision.transforms.functional.resize(tt, (100, 100))

        tt = tt[None, :, :, :]
        tt = torch.tensor(tt, dtype=torch.float) / 255.

        if tt.shape != (1, 3, 100, 100):
            continue

        image_tensor.append(tt)
        
        mask = image.replace('.jpg', '.png')
        dd = PIL.Image.open(f'/home/22061506mercedes/ComputerVision_UNET/data_flood/Mask{mask}')
        mm = torchvision.transforms.functional.pil_to_tensor(dd)
        mm = mm.repeat(3, 1, 1)
        mm = torchvision.transforms.functional.resize(mm, (100, 100))
        mm = mm[:1, :, :]

        mm = torch.tensor((mm > 0.).detach().numpy(), dtype=torch.long)
        mm = torch.nn.functional.one_hot(mm)
        mm = torch.permute(mm, (0, 3, 1, 2))
        mm = torch.tensor(mm, dtype=torch.float)

        masks_tensor.append(mm)
        
        image_tensor = torch.cat(image_tensor)
    print(image_tensor.shape)

    masks_tensor = torch.cat(masks_tensor)
    print(masks_tensor.shape)

    unet = UNet(n_channels=3, n_classes=2)

    dataloader_train_image = torch.utils.data.DataLoader(image_tensor, batch_size=64)
    dataloader_train_target = torch.utils.data.DataLoader(masks_tensor, batch_size=64)

    optim = torch.optim.Adam(unet.parameters(), lr=1e-3)
    cross_entropy = torch.nn.CrossEntropyLoss()
    
    loss_list = list()
    jaccard_list = list()
    for epoch in range(10):
        running_loss = 0.
        unet.train()

        jaccard_epoch = list()
        for image, target in zip(dataloader_train_image, dataloader_train_target):
            pred = unet(image)

            loss = cross_entropy(pred, target)
            running_loss += loss.item()

            loss.backward()
            optim.step()
    
    
        for image, target in zip(dataloader_train_image, dataloader_train_target):
            pred = unet(image)

            _, pred_unflatten = torch.max(pred, dim=1)
            _, target_unflatten = torch.max(target, dim=1)

            intersection = torch.sum(pred_unflatten == target_unflatten, dim=(1, 2)) / 10000.

            jaccard_epoch.append(torch.mean(intersection).detach())

    jaccard_list.append(sum(jaccard_epoch) / len(jaccard_epoch))
    loss_list.append(running_loss)
    
    

    