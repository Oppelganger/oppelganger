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
ADD https://yaso.su/wav2lipgan models/wav2lip.pth
ADD https://yaso.su/blazefaceshort models/blazeface.tflite
ADD https://yaso.su/gfpgan models/gfpgan.pth
COPY ./wav2lip/ .
COPY ./main.py .
COPY ./launch.sh /
EXPOSE 6873/tcp
ENTRYPOINT [ "/launch.sh" ]
