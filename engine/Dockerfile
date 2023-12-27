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
		torch ;\
	pip cache purge

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS final
RUN set -eux ;\
	apt update ;\
	apt -y full-upgrade ;\
	apt -y install python3.11 ;\
	apt clean
WORKDIR /work
COPY --from=prepare /work/venv venv
ADD https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q6_K.gguf?download=true models/
COPY main.py .
COPY launch.sh /

ENTRYPOINT [ "/launch.sh" ]
