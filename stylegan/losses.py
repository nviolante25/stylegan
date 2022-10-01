import torch.nn.functional as F
from torch.autograd import grad


def generator_loss(fake_logits):
    """Returns non-staurating loss for the generator

    The original generator loss log(1 - D(fake)) suffers from vanishing
    gradients when D(fake)=0 (zero loss).
    The non-staurating loss -log D(fake) guarantees strong gradients.

    Trick:
        - softplus(x) := log(1 + exp(x))
        - -log sigmoid(fake_logit) = softplus(-fake_logit)
    """
    return F.softplus(-fake_logits).mean()


def discriminator_loss(real_logits, fake_logits):
    """Returns discriminator loss: -log D(real) - log(1 - D(fake)).

    Trick:
        - softplus(x) := log(1 + exp(x))
        - -log sigmoid(real_logit) = softplus(-real_logit)
        - -log(1 - sigmoid(fake_logit)) = softplus(fake_logit)
    """
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def r1_loss(real_logits, real_images):

    # Need to set create_graph=True to backpropagate the loss afterwards
    discriminator_grad = grad(outputs=real_logits.sum(), inputs=real_images, create_graph=True)[0]
    batch_size = discriminator_grad.shape[0]
    r1_penalty = (discriminator_grad.view(batch_size, -1) ** 2).sum(1).mean()
    return r1_penalty
