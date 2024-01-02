import torch
from torch import nn


class Conv2d(nn.Module):
	conv_block: nn.Sequential
	act: nn.ReLU
	residual: bool

	def __init__(
		self,
		cin: int,
		cout: int,
		kernel_size: int,
		stride: int | tuple[int, int],
		padding: int,
		residual: bool = False
	):
		super().__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(cin, cout, kernel_size, stride, padding),
			nn.BatchNorm2d(cout)
		)
		self.act = nn.ReLU()
		self.residual = residual

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.conv_block(x)
		if self.residual:
			out += x
		return self.act(out)


class NoNormConv2d(nn.Module):
	def __init__(
		self, cin, cout, kernel_size, stride, padding
	):
		super().__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(cin, cout, kernel_size, stride, padding),
		)
		self.act = nn.LeakyReLU(0.01, inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.conv_block(x)
		return self.act(out)


class Conv2dTranspose(nn.Module):
	conv_block: nn.Sequential
	act: nn.ReLU

	def __init__(
		self,
		cin: int,
		cout: int,
		kernel_size: int,
		stride: int,
		padding: int,
		output_padding: int = 0,
	):
		super().__init__()
		self.conv_block = nn.Sequential(
			nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
			nn.BatchNorm2d(cout),
		)
		self.act = nn.ReLU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.conv_block(x)
		return self.act(out)
