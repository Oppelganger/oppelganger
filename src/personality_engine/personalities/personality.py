from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Personality:
	path: Path

	prompt: str
	sample_video: list[str]
	sample_audio: list[str]
	enhance: bool
	female: bool

	gpt_cond_latent: torch.Tensor
	speaker_embedding: torch.Tensor
