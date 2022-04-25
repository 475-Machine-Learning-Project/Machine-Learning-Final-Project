import torchvision

imagenet_data = torchvision.datasets.Places365('./Data','train-standard',small=True,download=True)