ARG CUDA_VERSION

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS prepare
WORKDIR /work

RUN set -eux ;\
	apt update ;\
	apt -y full-upgrade ;\
	apt -y install \
		python3.11-venv \
		python3.11-dev \
		build-essential \
		libgl1 \
		libglib2.0-0 ;\
	python3.11 -m venv venv ;\
	. venv/bin/activate ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install torch torchvision ;\
	pip install \
		numpy \
		basicsr \
		lmdb \
		opencv-python \
		pyyaml \
		SciPy \
		tb-nightly \
		tqdm \
		yapf ;\
	pip install --no-dependencies facexlib ;\
	pip install \
		librosa \
		mediapipe \
		gfpgan \
		"fastapi[all]" ;\
	pip cache purge

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS final
WORKDIR /work
RUN set -eux ;\
	apt update ;\
	apt -y full-upgrade ;\
	apt -y install \
		python3.11 \
		ffmpeg \
		libgl1 \
		libglib2.0-0 ;\
	apt clean
COPY --from=prepare /work/venv venv/
COPY ./wav2lip.pth models/
COPY ./blazeface.tflite models/
COPY ./gfpgan.pth models/
COPY ./wav2lip/ wav2lip/
COPY ./main.py .
COPY ./launch.sh /
EXPOSE 6873/tcp
ENTRYPOINT [ "/launch.sh" ]
