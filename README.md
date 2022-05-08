# Machine-Learning-Final-Project

Our final project tackles the task of image inpainting, which refers to the challenge of repairing missing or damaged portions of an image in such a way that the generated image closely resembles the original image. As an example, in the images below, an inpainting solution would attempt to fill in the white squares to best reproduce an input that can plausibly pass for the original image. To achieve this task, we will use a Generative Adversarial Network (GAN).

## Setup
```sh
git clone https://github.com/475-Machine-Learning-Project/Machine-Learning-Final-Project.git
pip install requirements.txt
```

## Running the code
- Baseline methods (Navier Stokes & Fast Marching): AnalyticalSolns.ipynb
- GAN model training, development, and testing: tfgan/model.ipynb
