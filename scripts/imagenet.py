#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import tnn.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms.v2 as v2
import torch.utils.data as data

from tnn.resnet import ResNet
from datasets import load_dataset


# In[13]:


with open("./.token.txt", mode="r") as file:
    token = file.readline().strip()

imagenet = load_dataset(
    "ILSVRC/imagenet-1k", token=token, num_proc=1, trust_remote_code=True
)


# In[14]:


train, val = imagenet.get("train"), imagenet.get("validation")
train = train.select(np.random.choice(1000, 1000, replace=False))
val = val.select(np.random.choice(1000, 1000, replace=False))


# In[15]:


def pre_process(example):
    transforms = v2.Compose(
        [
            v2.Resize(256),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    example["input"] = transforms(example["image"])
    return example


train = train.map(pre_process, batched=True, batch_size=256, num_proc=1)
val = val.map(pre_process, batched=True, batch_size=256, num_proc=1)

train.set_format(type="pt", columns=["input", "label"], output_all_columns=True)
val.set_format(type="pt", columns=["input", "label"], output_all_columns=True)


# In[39]:


def collate_fn(batch):
    augmentor = v2.Compose(
        [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(size=(224, 224))]
    )
    batch_size = len(batch)
    inputs = [batch[i]["input"] for i in range(batch_size)]
    labels = torch.tensor([batch[i]["label"] for i in range(batch_size)]).long()
    inputs = [augmentor(i) for i in inputs]
    inputs = torch.stack(inputs, dim=0)
    return inputs, labels


batch_size = 256
drop_last = False
shuffle = True
train_loader = data.DataLoader(
    train,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    collate_fn=collate_fn,
    num_workers=4,
)


# In[7]:


layer_config = [(64, 64, 3), (64, 128, 4), (128, 256, 5), (256, 512, 3)]
model = ResNet(layer_config)
