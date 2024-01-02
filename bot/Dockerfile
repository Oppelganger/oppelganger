FROM docker.io/ubuntu:22.04 AS prepare
WORKDIR /work
RUN --mount=type=cache,target=/root/.cache/pip \
	set -eux ;\
	apt update ;\
	apt -y full-upgrade ;\
	apt -y install python3.11-venv ;\
	python3.11 -m venv venv ;\
	. venv/bin/activate ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install \
		pyrogram \
		tgcrypto \
		uvloop \
		aiohttp ;\
	pip cache purge

FROM docker.io/ubuntu:22.04 AS final
WORKDIR /work
RUN set -eux ;\
	apt update ;\
	apt -y full-upgrade ;\
	apt -y install python3.11 ;\
	apt clean
COPY --from=prepare /work/venv venv
COPY main.py .
COPY launch.sh /

ENTRYPOINT [ "/launch.sh" ]
