from dataclasses import dataclass

import torch

from .types import Language


@dataclass
class Personality:
	id: str
	prompt: str
	video_objects: list[str]
	audio_objects: list[str]
	language: Language
	enhance: bool
	female: bool

	gpt_cond_latent: torch.Tensor
	speaker_embedding: torch.Tensor
