from os import PathLike
from typing import TypedDict, Mapping, Optional, Any

import torch

from .models import Wav2Lip


def load_wav2lip(
	device: str | torch.device,
	path: str | PathLike = '/models/wav2lip.pth'
):
	model = Wav2Lip()

	class Wav2LipCheckpoint(TypedDict):
		state_dict: Mapping[str, Any]
		optimizer: Optional[Mapping[str, Any]]
		global_step: int
		global_epoch: int

	checkpoint: Wav2LipCheckpoint = torch.load(path, map_location=torch.device(device))
	state_dict = checkpoint['state_dict']
	new_state_dict = {}
	for key, value in state_dict.items():
		new_state_dict[key.replace('module.', '')] = value
	model.load_state_dict(new_state_dict)

	return model.to(device).eval()
