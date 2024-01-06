from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import torch
from gfpgan import GFPGANer

from .wav2lip import (
	wav2lip as wav2lip_sync,
	load_wav2lip as load_wav2lip_sync,
	Wav2Lip as Wav2LipMode,
)
from .._gfpgan import load_gfpgan
from ..blazeface import BlazeFace, load_blazeface
from ..utils import awaitable

load_wav2lip = awaitable(load_wav2lip_sync)
wav2lip = awaitable(wav2lip_sync)


@dataclass
class Wav2Lip:
	device: torch.device
	model: Wav2LipMode
	blazeface: BlazeFace
	gfpgan_model: GFPGANer

	@classmethod
	async def load(
		cls,
		device: str | torch.device,
		path: str | PathLike = '/models/wav2lip.pth',
		blazeface_path: str | PathLike = '/models/blazeface.tflite',
		gfpgan_path: str | PathLike = '/models/gfpgan.pth',
	) -> Wav2Lip:
		return cls(
			torch.device(device),
			await load_wav2lip(device, path),
			await load_blazeface(blazeface_path),
			await load_gfpgan(device, gfpgan_path)
		)

	async def process(
		self,
		in_audio: str,
		in_video: str,
		out_video: str,
		enhance: bool,
		female: bool
	):
		return await wav2lip(
			self.device,
			self.model,
			self.blazeface,
			self.gfpgan_model,
			in_audio,
			in_video,
			out_video,
			enhance,
			female
		)
