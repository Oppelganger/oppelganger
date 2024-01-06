import os
import subprocess
import uuid
from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import torch
from gfpgan import GFPGANer
from mediapipe.tasks.python import vision as mpv

from .audio import load_wav, melspectrogram
from .models import Wav2Lip

face_size: int = 96
batch_size: int = 128
mel_step_size: int = 16
fourcc: int = cv2.VideoWriter.fourcc(*'I420')  # yuv420p


def get_smoothened_boxes(boxes: np.ndarray, val: int) -> np.ndarray:
	for i in range(len(boxes)):
		if i + val > len(boxes):
			window = boxes[len(boxes) - val:]
		else:
			window = boxes[i: i + val]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def face_detect(
	blazeface: mpv.FaceDetector,
	images: List[cv2.typing.MatLike]
) -> List[tuple[cv2.typing.MatLike, cv2.typing.Size]]:
	results: List[cv2.typing.Size] = []

	for image in images:
		image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		detections = blazeface.detect(image).detections

		if len(detections) < 1:
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		detection = detections[0].bounding_box

		x1: int = detection.origin_x
		y1: int = detection.origin_y
		x2: int = detection.origin_x + detection.width
		y2: int = detection.origin_y + detection.height

		results.append((x1, y1, x2, y2))

	boxes = get_smoothened_boxes(np.array(results), val=5)

	return [
		(image[y1:y2, x1:x2], (y1, y2, x1, x2))
		for image, (x1, y1, x2, y2) in zip(images, boxes)
	]


def datagen(
	frames: List[cv2.typing.MatLike],
	mels: List[np.ndarray[np.floating]],
	faces: List[tuple[cv2.typing.MatLike, cv2.typing.Size]]
):
	img_batch: List[cv2.typing.MatLike] = []
	mel_batch = []

	frame_batch: List[cv2.typing.MatLike] = []
	coords_batch: List[cv2.typing.Size] = []

	for i, m in enumerate(mels):
		idx = i % len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = faces[idx]

		face = cv2.resize(face, (face_size, face_size))

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= batch_size:
			img_batch, mel_batch = datagen_inner(img_batch, mel_batch)
			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = datagen_inner(img_batch, mel_batch)
		yield img_batch, mel_batch, frame_batch, coords_batch


def datagen_inner(img_batch, mel_batch):
	img_batch = np.asarray(img_batch)
	mel_batch = np.asarray(mel_batch)

	img_masked = img_batch.copy()
	img_masked[:, face_size // 2:] = 0

	np_img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
	np_mel_batch = np.reshape(
		mel_batch,
		[
			len(mel_batch),
			mel_batch.shape[1],
			mel_batch.shape[2],
			1
		]
	)

	return np_img_batch, np_mel_batch


@dataclass
class VideoCache:
	fps: float
	frames: List[cv2.typing.MatLike]
	faces: List[tuple[cv2.typing.MatLike, cv2.typing.Size]]


video_cache = {}


def wav2lip(
	device: str | torch.device,
	model: Wav2Lip,
	blazeface: mpv.FaceDetector,
	gfpgan_model: GFPGANer,
	in_audio: str,
	in_video: str,
	out_video: str,
	enhance: bool,
	female: bool
):
	tmp_out = f'/tmp/result-{uuid.uuid4()}.nut'

	if in_video in video_cache:
		print('Using cached video')
		val = video_cache[in_video]
		fps = val.fps
		full_frames = val.frames
		faces = val.faces
	else:
		video_stream = cv2.VideoCapture(in_video)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Start reading video frames')

		full_frames = []

		while True:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			full_frames.append(frame)
		print('End reading video frames')
		print('Start detecting face on frames')
		faces = face_detect(blazeface, full_frames)
		print('End detecting face on frames')
		video_cache[in_video] = VideoCache(fps, full_frames, faces)

	wav = load_wav(in_audio)
	mel = melspectrogram(wav, female)

	mel_chunks = []
	mel_idx_multiplier = 80.0 / fps

	print('Start generating mel chunks')
	idx = 0
	while True:
		start_idx = int(idx * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
		idx += 1
	print('End generating mel chunks')

	print('Length of mel chunks: {}'.format(len(mel_chunks)))

	full_frames = full_frames[: len(mel_chunks)]

	print('Start datagen')
	gen = datagen(full_frames, mel_chunks, faces)
	print('End datagen')

	frame_h, frame_w = full_frames[0].shape[:-1]
	out = cv2.VideoWriter(tmp_out, fourcc, fps, (frame_w, frame_h))

	with torch.no_grad():
		for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

			preds = model(mel_batch, img_batch)
			preds = preds.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

			for pred, frame, coord in zip(preds, frames, coords):
				y1, y2, x1, x2 = coord

				if enhance:
					_, _, pred = gfpgan_model.enhance(pred)

				pred = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2 - y1))

				frame[y1:y2, x1:x2] = pred
				out.write(frame)

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -c:v h264_nvenc -preset fast {}'
	subprocess.call(command.format(in_audio, tmp_out, out_video), shell=True)
	os.remove(tmp_out)
