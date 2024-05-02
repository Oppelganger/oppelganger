from os import PathLike

from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision as mpv

BlazeFace = mpv.FaceDetector


def load_blazeface(path: str | PathLike = '/models/blazeface.tflite') -> BlazeFace:
	base_options = mpp.BaseOptions(model_asset_path=str(path))
	options = mpv.FaceDetectorOptions(base_options=base_options)
	detector = mpv.FaceDetector.create_from_options(options)

	return detector
