from model import TooSimpleDiscriminator, TooSimpleGenerator

def loss(model_g, model_d, x, z):
    """
    Calculate Wasserstein loss given models, real image, and noisy image
    
    Args:
        model_g (pytorch model): Generator model class object.
        model_g (pytorch model): Discriminator model class object.
        x (np array): Real image
        z (np array): Noisy image

    Returns:
        Scalar Wasserstein loss, between 0 and 1 inclusive
    """
    return abs(model_d.forward(x) - model_d.forward(model_g.forward(z)))
