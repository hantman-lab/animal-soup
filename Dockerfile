# Use nvidia/cuda image
FROM --platform=linux/amd64 nvidia/cuda:12.2.0-base-ubuntu20.04
FROM python:3.11

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt upgrade -y && apt-get install -y \
    curl \
    sudo \
    git \
    mesa-vulkan-drivers \
    bash-completion \
    libturbojpeg0-dev
#    ffmpeg libsm6 libxext6 libxcb-icccm4 \
#    libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb-dev libxkbcommon-x11-0 \
#    libpulse-mainloop-glib0 ubuntu-restricted-extras libqt5multimedia5-plugins vlc \

RUN pip install setuptools --upgrade && pip install --upgrade pip

# optional dependency of fpl that is useful
RUN pip install simplejpeg

# install repo
RUN git clone https://github.com/hantman-lab/animal-soup.git
WORKDIR animal-soup/
RUN pip install -e .

# download the models and sample data
RUN python -m animal_soup/utils/downloads

# Set the entrypoint
ENTRYPOINT [ "/bin/bash" ]
CMD ["pytest", "-s", "tests/"]
