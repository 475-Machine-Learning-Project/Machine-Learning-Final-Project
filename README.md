# Image inpainting with GANs

## Instructions

1) To install all dependencies, run `pip install pytorch && pip install Pillow && pip install torchvision
2) To download the dataset, run the `DownloadData.py` file. This will download the MIT Places Dataset using torchvision's dataset loader.
3) To create masked images, run the `MakeData.py` file. This will preprocess the images by adding a mask at the center of the screen.
4) To run the baseline analytical inpainting solutions, follow the instructions on the `inpaint.ipynb` file.
5) To run the GAN inpainting solution, follow the instructions on the `inpaint.ipynb` file.
