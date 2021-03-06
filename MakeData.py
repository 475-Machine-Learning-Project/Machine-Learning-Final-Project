
from PIL import Image
import random

from torch import rand

for index in range(1, 5001):
# for index in range(1, 100):
    

    # Import an image from directory:
    input_image = Image.open("Dataset/0/original/" + f'{index:08}' + ".jpg")
    
    # Extracting pixel map:
    pixel_map = input_image.load()
    
    # Extracting the width and height 
    # of the image:
    width, height = input_image.size
    # print(width, height)

    # Create an image as input:
    image_mask = Image.new(mode="RGB", size=(width, height),
                            color="white")
    mask_map = image_mask.load()

    # Generate random mask:
    # startx = random.randint(0, width)
    # starty = random.randint(0, height)

    # lengthx = random.randint(31, 81)
    # lengthy = random.randint(31, 81)

    # Generate mask at center, 1.5% of image:
    lengthx = int((width * (0.015 ** 0.5)))
    lengthy = int((height * (0.015 ** 0.5)))

    startx = width/2 - (0.5 * lengthx)
    starty = height/2 - (0.5 * lengthy)

    if startx + lengthx > width:
        lengthx = width - startx

    if starty + lengthy > height:
        lengthy = height - starty



    # taking half of the width:
    for i in range(lengthx):
        for j in range(lengthy):
            
            # setting the pixel value.
            pixel_map[startx + i,starty + j] = (int(256), int(256), int(256))

            mask_map[startx + i,starty + j] = (int(256), int(256), int(256))
    
    # Saving the final output
    # as "grayscale.png":
    input_image.save("Dataset/0/images-center/"+f'{index:08}'+".png", format="png")

    if index == 1:
        image_mask.save("Dataset/0/masks-center/"+f'{index:08}'+".png", format="png")
    
    # use input_image.show() to see the image on the
    # output screen.

print("Done!")