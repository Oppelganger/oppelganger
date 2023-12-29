ARG CUDA_VERSION

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS prepare
WORKDIR /work
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN --mount=type=cache,target=/root/.cache/pip \
	set -eux ;\
	apt update ;\
	apt -y full-upgrade ;\
	apt -y install python3.11-venv ;\
	python3.11 -m venv venv ;\
	. venv/bin/activate ;\
	pip install --upgrade setuptools wheel ;\
	pip install setuptools wheel ;\
	pip install \
		TTS \
		llama-cpp-python \
		dotmap \
		torch \
		"fastapi[all]" ;\
	pip cache purge

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS final
WORKDIR /work
ADD https://huggingface.co/TheBloke/Mixtral_7Bx2_MoE-GGUF/resolve/main/mixtral_7bx2_moe.Q5_K_M.gguf models/
RUN set -eux ;\
	apt update ;\
	apt -y full-upgrade ;\
	apt -y install python3.11 ffmpeg ;\
	apt clean
COPY --from=prepare /work/venv venv
COPY main.py .
COPY launch.sh /

ENTRYPOINT [ "/launch.sh" ]
