from dataclasses import dataclass

import torch


@dataclass
class Personality:
	id: str
	prompt: str
	video_objects: list[str]
	audio_objects: list[str]
	enhance: bool
	female: bool

	gpt_cond_latent: torch.Tensor
	speaker_embedding: torch.Tensor
