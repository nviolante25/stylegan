import torch.nn.functional as F


def generator_loss(fake_logit):
    """Returns non-staurating loss for the generator

    The original generator loss log(1 - D(fake)) suffers from vanishing 
    gradients when D(fake)=0 (zero loss).
    The non-staurating loss -log D(fake) guarantees strong gradients.

    Trick:
        - softplus(x) := log(1 + exp(x))
        - -log sigmoid(fake_logit) = softplus(-fake_logit)
    """
    return F.softplus(-fake_logit).mean()


def discriminator_loss(real_logit, fake_logit):
    """Returns discriminator loss: -log D(real) - log(1 - D(fake)).

    Trick:
        - softplus(x) := log(1 + exp(x))
        - -log sigmoid(real_logit) = softplus(-real_logit)
        - -log(1 - sigmoid(fake_logit)) = softplus(fake_logit)
    """
    return F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
