ARG CUDA_VERSION

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder
WORKDIR /work

ARG PYTHON=3.11

RUN --mount=type=cache,target=/root/.cache/pdm \
	set -eux ;\
	export DEBIAN_FRONTEND=noninteractive ;\
	\
	apt update ;\
	apt -y full-upgrade ;\
  apt -y install software-properties-common ;\
  add-apt-repository -y ppa:deadsnakes/ppa ;\
	apt -y install \
    build-essential \
    python${PYTHON}-dev \
    python3-pip \
    libgl1 \
		libglib2.0-0 ;\
	python${PYTHON} -m pip install -U pip setuptools wheel ;\
	python${PYTHON} -m pip install pdm

COPY pyproject.toml pdm.lock ./
RUN set -eux ;\
	export CMAKE_ARGS="-DLLAMA_CUBLAS=on" ;\
	\
	mkdir __pypackages__ ;\
	pdm sync --prod --no-editable

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS final
WORKDIR /work

ARG PYTHON=3.11
ENV PYTHON_EXE=python${PYTHON}

RUN set -eux ;\
	export DEBIAN_FRONTEND=noninteractive ;\
	\
	apt update ;\
	apt -y full-upgrade ;\
  apt -y install software-properties-common ;\
  add-apt-repository -y ppa:deadsnakes/ppa ;\
	apt -y install \
		python${PYTHON} \
    ffmpeg \
    libgl1 \
    libglib2.0-0 ;\
	apt -y purge software-properties-common ;\
	apt -y autoremove ;\
	apt clean

ENV PYTHONPATH=/work/pkgs
COPY --from=builder /work/__pypackages__/${PYTHON}/lib ./pkgs

ARG LLM_URL=https://huggingface.co/TheBloke/Mixtral_7Bx2_MoE-GGUF/resolve/main/mixtral_7bx2_moe.Q5_K_M.gguf
ADD ${LLM_URL} /models/llm.gguf

ARG XTTS_BASE_URL=https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main
ADD ${XTTS_BASE_URL}/config.json /models/xtts/
ADD ${XTTS_BASE_URL}/vocab.json /models/xtts/
ADD ${XTTS_BASE_URL}/model.pth /models/xtts/
ADD ${XTTS_BASE_URL}/speakers_xtts.pth /models/xtts/
ADD ${XTTS_BASE_URL}/hash.md5 /models/xtts/

COPY models /models
COPY personalities /personalities

COPY src/launch.sh /launch.sh
COPY src/personality_engine ./personality_engine


ENTRYPOINT [ "/launch.sh" ]
