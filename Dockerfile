FROM python:3.10-bookworm

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        python3-pip \
        libgtk-3-bin \
        squashfs-tools \
        libglib2.0-bin \
        fakeroot

# Credit to perrito666 for the ubuntu fix
RUN python3 -m pip install git+https://github.com/hexbabe/appimage-builder.git

WORKDIR /app

COPY . /app

COPY packaging/. /app

CMD ["appimage-builder"]
