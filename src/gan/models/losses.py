# src/gan/models/losses.py

import torch
import torch.autograd as autograd
from typing import Tuple


# --------------------------------------------------
# Generator loss (Wasserstein)
# --------------------------------------------------
def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Generator wants discriminator to think fake images are real
    """
    return -fake_scores.mean()


# --------------------------------------------------
# Critic Wasserstein loss
# --------------------------------------------------
def critic_wasserstein_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    """
    WGAN critic loss (no sigmoid)
    """
    return fake_scores.mean() - real_scores.mean()


# --------------------------------------------------
# Gradient Penalty (WGAN-GP)
# --------------------------------------------------
def gradient_penalty(
    discriminator,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    labels: torch.Tensor = None,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Enforces Lipschitz constraint for WGAN-GP
    """

    batch_size = real_images.size(0)
    device = real_images.device

    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)

    # Discriminator output
    scores = (
        discriminator(interpolated, labels)
        if labels is not None
        else discriminator(interpolated)
    )

    # Gradient w.r.t. interpolated images
    gradients = autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)

    gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return gp


# --------------------------------------------------
# Full critic loss with GP
# --------------------------------------------------
def critic_loss_with_gp(
    discriminator,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    labels: torch.Tensor = None,
    lambda_gp: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        total_loss: critic loss + gradient penalty
        w_loss: wasserstein loss (detached)
        gp: gradient penalty (detached)
    """

    fake_images = fake_images.detach()

    real_scores = (
        discriminator(real_images, labels)
        if labels is not None
        else discriminator(real_images)
    )

    fake_scores = (
        discriminator(fake_images, labels)
        if labels is not None
        else discriminator(fake_images)
    )

    w_loss = critic_wasserstein_loss(real_scores, fake_scores)
    gp = gradient_penalty(
        discriminator,
        real_images,
        fake_images,
        labels,
        lambda_gp,
    )

    total_loss = w_loss + gp
    return total_loss, w_loss.detach(), gp.detach()
