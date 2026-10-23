from __future__ import annotations

import math
from typing import Any

from geoforge_diffusion.models.edm import EDMLoss, EDMLossConfig, EDMPreconditioner
from geoforge_diffusion.models.unet import MultiResolutionUNet


def create_edm_model(config: dict[str, Any]) -> EDMPreconditioner:
    model_config = dict(config["model"])
    network = MultiResolutionUNet(**model_config)
    return EDMPreconditioner(network, sigma_data=float(config["edm"]["sigma_data"]))


def create_edm_loss(config: dict[str, Any]) -> EDMLoss:
    loss_config = EDMLossConfig(**config["edm"]["loss"])
    model_sigma_data = float(config["edm"]["sigma_data"])
    if not math.isclose(loss_config.sigma_data, model_sigma_data, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("edm.sigma_data and edm.loss.sigma_data must match")
    return EDMLoss(loss_config)
