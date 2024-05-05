from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import torch
from gfpgan import GFPGANer

from .wav2lip import (
	wav2lip,
	load_wav2lip,
	Wav2Lip as Wav2LipMode,
)
from .._gfpgan import load_gfpgan
from ..blazeface import BlazeFace, load_blazeface
from ..handler.stage import Stage


@dataclass
class Wav2Lip:
	device: torch.device
	model: Wav2LipMode
	blazeface: BlazeFace
	gfpgan_model: GFPGANer

	@classmethod
	def load(
		cls,
		device: str | torch.device,
		path: str | PathLike = '/models/wav2lip.pth',
		blazeface_path: str | PathLike = '/models/blazeface.tflite',
		gfpgan_path: str | PathLike = '/models/gfpgan.pth',
	) -> Wav2Lip:
		return cls(
			torch.device(device),
			load_wav2lip(device, path),
			load_blazeface(blazeface_path),
			load_gfpgan(device, gfpgan_path)
		)

	def process(
		self,
		in_audio: str,
		in_video: str,
		out_video: str,
		enhance: bool,
		female: bool,
		stage: Stage
	):
		return wav2lip(
			self.device,
			self.model,
			self.blazeface,
			self.gfpgan_model,
			in_audio,
			in_video,
			out_video,
			enhance,
			female,
			stage
		)
