# from model import TooSimpleDiscriminator, TooSimpleGenerator
# import argparse
import os
# import random
import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torchvision.utils as vutils
import numpy as np

print('hello world')

# # class TooSimpleGenerator(torch.nn.Module):

# #     def __init__(self, ):
# #         super().__init__()
# #         # TODO init params
# #         self.beta = []

# #     def forward(self, X):
# #         # TODO
# #         return X

# # class TooSimpleDiscriminator(torch.nn.Module):

# #     def __init__(self):
# #         super().__init__()
# #         # TODO init params

# #     def forward(self, X):
# #         # TODO
# #         return X

# ###########

# print('hello')

# epochs = 0
# batch_size = 0
# lr = 0
# # additional params go here
# beta1_D = 0
# beta1_G = 0

# generator = TooSimpleGenerator()
# discriminator = TooSimpleDiscriminator()

# print('here')

# # Load original and masked images



# # dataset = dset.ImageFolder('../Dataset/0/original/', transform=transforms.ToTensor())
# # print(dataset.shape)
# # print(type(dataset))

# optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1_G, 0.999))
# optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1_D, 0.999))


# for epoch in range(epochs):
#     break
#     #for i in data_dir
#     # TODO: need to train discriminator on real data

#     # TODO: generate fake data

#     # TODO: train discriminator on fake data

#     # TODO: train generator
