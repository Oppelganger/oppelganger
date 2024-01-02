ARG CUDA_VERSION

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS prepare
WORKDIR /work

RUN set -eux ;\
	apt update ;\
	apt -y full-upgrade ;\
	apt -y install python3.11-venv python3.11-dev build-essential ;\
	python3.11 -m venv venv ;\
	. venv/bin/activate ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install \
		numpy \
    SciPy \
		librosa \
    mediapipe \
		opencv-python \
    gfpgan \
		torch==1.13.1 \
		"fastapi[all]" ;\
	pip cache purge

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS final
WORKDIR /work
RUN set -eux ;\
	apt update ;\
	apt -y full-upgrade ;\
	apt -y install python3.11 ffmpeg ;\
	apt clean
COPY --from=prepare /work/venv venv
COPY wav2lip.pth models/
COPY blazeface.tflite models/
COPY gfpgan.pth models/
COPY ./wav2lip/ .
COPY ./main.py .
COPY ./launch.sh /
EXPOSE 6873/tcp
ENTRYPOINT [ "/launch.sh" ]
