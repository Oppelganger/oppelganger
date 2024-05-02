from os import PathLike

import torch
from gfpgan import GFPGANer


def load_gfpgan(
	device: str | torch.device,
	path: str | PathLike = '/models/gfpgan.pth',
) -> GFPGANer:
	return GFPGANer(
		model_path=path,
		device=device
	)
