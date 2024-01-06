from os import PathLike

import torch
from gfpgan import GFPGANer

from ..utils import awaitable


@awaitable
def load_gfpgan(
	device: str | torch.device,
	path: str | PathLike = '/models/gfpgan.pth',
) -> GFPGANer:
	return GFPGANer(
		model_path=path,
		device=device
	)
