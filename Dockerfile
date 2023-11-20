FROM python:3.9-bookworm

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        python3-pip \
        libgtk-3-bin \
        squashfs-tools \
        zsync \
        libglib2.0-bin \
        fakeroot

RUN pip3 install appimage-builder

WORKDIR /app

COPY . /app

COPY packaging/. /app

CMD ["appimage-builder"]
