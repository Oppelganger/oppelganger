from os import PathLike

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def load_xtts(path: str | PathLike = "/models/xtts") -> Xtts:  # TODO: device
	config = XttsConfig()
	config.load_json(f"{path}/config.json")
	xtts = Xtts.init_from_config(config)
	xtts.load_checkpoint(
		config,
		checkpoint_dir=str(path),
		vocab_path=f"{path}/vocab.json",
		use_deepspeed=False
	)

	if torch.cuda.is_available():
		xtts = xtts.cuda()

	return xtts


def inference(
	model: Xtts,
	text: str,
	gpt_cond_latent: torch.Tensor,
	speaker_embedding: torch.Tensor,
	path: str | PathLike
):
	res = model.inference(
		text,
		"en",
		gpt_cond_latent,
		speaker_embedding,
		speed=1.1,
		enable_text_splitting=True,
		temperature=model.config.temperature,
		length_penalty=model.config.length_penalty,
		repetition_penalty=model.config.repetition_penalty,
		top_k=model.config.top_k,
		top_p=model.config.top_p,
	)
	# noinspection PyUnresolvedReferences
	torchaudio.save(str(path), torch.tensor(res["wav"]).unsqueeze(0), 24000)
