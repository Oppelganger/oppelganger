import torch
from torch import nn

from .conv import Conv2dTranspose, Conv2d, NoNormConv2d


class Wav2Lip(nn.Module):
	face_encoder_blocks: nn.ModuleList
	audio_encoder: nn.Sequential
	face_decoder_blocks: nn.ModuleList
	output_block: nn.Sequential

	def __init__(self):
		super(Wav2Lip, self).__init__()

		self.face_encoder_blocks = nn.ModuleList(
			[
				nn.Sequential(
					Conv2d(6, 16, kernel_size=7, stride=1, padding=3)
				),  # 96,96
				nn.Sequential(
					Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
					Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
					Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 48,48
				nn.Sequential(
					Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
					Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
					Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
					Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 24,24
				nn.Sequential(
					Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
					Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
					Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 12,12
				nn.Sequential(
					Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
					Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
					Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 6,6
				nn.Sequential(
					Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
					Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 3,3
				nn.Sequential(
					Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
					Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
				),  # 1, 1
			]
		)

		self.audio_encoder = nn.Sequential(
			Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
			Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
			Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
			Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
			Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
			Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
			Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
		)

		self.face_decoder_blocks = nn.ModuleList(
			[
				nn.Sequential(
					Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
				),
				nn.Sequential(
					Conv2dTranspose(
						1024, 512, kernel_size=3, stride=1, padding=0
					),
					Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 3,3
				nn.Sequential(
					Conv2dTranspose(
						1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
					),
					Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
					Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 6, 6
				nn.Sequential(
					Conv2dTranspose(
						768, 384, kernel_size=3, stride=2, padding=1, output_padding=1
					),
					Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
					Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 12, 12
				nn.Sequential(
					Conv2dTranspose(
						512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
					),
					Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
					Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 24, 24
				nn.Sequential(
					Conv2dTranspose(
						320, 128, kernel_size=3, stride=2, padding=1, output_padding=1
					),
					Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
					Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 48, 48
				nn.Sequential(
					Conv2dTranspose(
						160, 64, kernel_size=3, stride=2, padding=1, output_padding=1
					),
					Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
					Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
				),  # 96,96
			]
		)

		self.output_block = nn.Sequential(
			Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid(),
		)

	# noinspection PyPep8Naming
	def forward(self, audio_sequences, face_sequences):
		# audio_sequences = (B, T, 1, 80, 16)
		B = audio_sequences.size(0)

		input_dim_size = len(face_sequences.size())
		if input_dim_size > 4:
			audio_sequences = torch.cat(
				[
					audio_sequences[:, i]
					for i in range(audio_sequences.size(1))
				],
				dim=0
			)
			face_sequences = torch.cat(
				[
					face_sequences[:, :, i]
					for i in range(face_sequences.size(2))
				],
				dim=0
			)

		audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

		feats = []
		x = face_sequences
		for f in self.face_encoder_blocks:
			x = f(x)
			feats.append(x)

		x = audio_embedding
		for f in self.face_decoder_blocks:
			x = f(x)
			try:
				x = torch.cat((x, feats[-1]), dim=1)
			except Exception as e:
				print(x.size())
				print(feats[-1].size())
				raise e

			feats.pop()

		x = self.output_block(x)

		if input_dim_size > 4:
			x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
			outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

		else:
			outputs = x

		return outputs


class Wav2LipDiscQual(nn.Module):
	face_encoder_blocks: nn.ModuleList
	binary_pred: nn.Sequential
	label_noise: float

	def __init__(self):
		super(Wav2LipDiscQual, self).__init__()

		self.face_encoder_blocks = nn.ModuleList(
			[
				nn.Sequential(
					NoNormConv2d(3, 32, kernel_size=7, stride=1, padding=3)
				),  # 48,96
				nn.Sequential(
					NoNormConv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),
					NoNormConv2d(64, 64, kernel_size=5, stride=1, padding=2),
				),  # 48,48
				nn.Sequential(
					NoNormConv2d(64, 128, kernel_size=5, stride=2, padding=2),
					NoNormConv2d(128, 128, kernel_size=5, stride=1, padding=2),
				),  # 24,24
				nn.Sequential(
					NoNormConv2d(128, 256, kernel_size=5, stride=2, padding=2),
					NoNormConv2d(256, 256, kernel_size=5, stride=1, padding=2),
				),  # 12,12
				nn.Sequential(
					NoNormConv2d(256, 512, kernel_size=3, stride=2, padding=1),
					NoNormConv2d(512, 512, kernel_size=3, stride=1, padding=1),
				),  # 6,6
				nn.Sequential(
					NoNormConv2d(512, 512, kernel_size=3, stride=2, padding=1),
					NoNormConv2d(512, 512, kernel_size=3, stride=1, padding=1),
				),  # 3,3
				nn.Sequential(
					NoNormConv2d(512, 512, kernel_size=3, stride=1, padding=0),
					NoNormConv2d(512, 512, kernel_size=1, stride=1, padding=0),
				),  # 1, 1
			]
		)

		self.binary_pred = nn.Sequential(
			nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

		self.label_noise = 0.0

	@staticmethod
	def _get_lower_half(face_sequences):
		return face_sequences[:, :, face_sequences.size(2) // 2:]

	@staticmethod
	def _to_2d(face_sequences):
		face_sequences = torch.cat(
			[
				face_sequences[:, :, i]
				for i in range(face_sequences.size(2))
			],
			dim=0
		)
		return face_sequences

	def forward(
		self,
		face_sequences: torch.Tensor
	) -> torch.Tensor:
		face_sequences = self.to_2d(face_sequences)
		face_sequences = self._get_lower_half(face_sequences)

		x = face_sequences
		for f in self.face_encoder_blocks:
			x = f(x)

		return self.binary_pred(x).view(len(x), -1)
