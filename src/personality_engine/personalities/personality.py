from dataclasses import dataclass
from typing import Tuple

import torch

from .json import PersonalityJson


@dataclass
class Personality:
	id: str
	prompt: str
	video_objects: list[Tuple[str, Tuple[int, int, int, int]]]  # x1, y1, x2, y2
	audio_objects: list[str]
	enhance: bool
	female: bool

	gpt_cond_latent: torch.Tensor
	speaker_embedding: torch.Tensor
