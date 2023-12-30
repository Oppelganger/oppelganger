FROM docker.io/ubuntu:22.04 AS downloads
WORKDIR /work

ADD https://codeload.github.com/Kamillaova/Wav2Lip/tar.gz/5a3ec355be0f1c5a6e310f30fb752b5227c8aac0 .
ADD https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth .
ADD ./wav2lip.pth .

RUN set -eux ;\
	mkdir Wav2Lip ;\
	ls ;\
	tar xpf ./5a3ec355be0f1c5a6e310f30fb752b5227c8aac0 -C Wav2Lip --strip-components=1 ;\
	mv s3fd-619a316812.pth s3fd.pth

FROM docker.io/golang:1.21-alpine3.18 AS golang
WORKDIR /work
ENV CGO_ENABLED=0

COPY go.mod go.sum .
RUN --mount=type=cache,mode=0755,target=/go/pkg/mod go mod download
COPY main.go .
RUN --mount=type=cache,mode=0755,target=/go/pkg/mod \
	--mount=type=cache,target=/root/.cache/go-build \
	go build -ldflags="-w -s" -v -o main main.go

FROM docker.io/nvidia/cuda:11.7.1-runtime-ubuntu22.04 AS lipsync
WORKDIR /app

RUN set -eux ;\
	apt update ;\
	apt -y full-upgrade ;\
	apt -y install ffmpeg software-properties-common ;\
	apt-add-repository -y ppa:deadsnakes/ppa ;\
	apt -y remove --autoremove --purge software-properties-common ;\
	apt -y install python3.7-venv ;\
	python3.7 -m venv venv ;\
	apt clean

RUN set -eux ;\
	. venv/bin/activate ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install \
		librosa==0.7.0 \
		numpy==1.17.1 \
		opencv-python \
		tqdm==4.45.0 \
		numba==0.48 \
		torch==1.13.1 \
		torchvision==0.14.1 ;\
	pip cache purge

COPY --from=downloads /work/Wav2Lip .
COPY --from=downloads /work/s3fd.pth face_detection/detection/sfd
COPY --from=downloads /work/wav2lip.pth checkpoints

FROM lipsync AS final
WORKDIR /app
COPY --from=golang /work/main /main
COPY launch.sh /
EXPOSE 6873/tcp
ENTRYPOINT [ "/launch.sh" ]
