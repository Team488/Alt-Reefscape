FROM python:3.10.16-slim-bookworm

# Install basic dependencies, Python-related packages, and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates curl wget \
    openssl git ffmpeg tar lsb-release \
    procps manpages-dev unzip zip xauth swig \
    python3-numpy python3-distutils python3-setuptools python3-pyqt5 python3-opencv \
    libboost-python-dev libboost-thread-dev libatlas-base-dev libavcodec-dev \
    libavformat-dev libavutil-dev libcanberra-gtk3-module libeigen3-dev \
    libglew-dev libgl1-mesa-dev libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
    libgtk-3-dev libjpeg-dev liblapack-dev liblapacke-dev libopenblas-dev \
    libopencv-dev libpng-dev libpostproc-dev libpq-dev libsm6 libswscale-dev \
    libtbb-dev libtesseract-dev libtiff-dev libtiff5-dev libv4l-dev libx11-dev \
    libxext6 libxine2-dev libxrender-dev libxvidcore-dev libx264-dev \
    libgtkglext1 libgtkglext1-dev libvtk9-dev libdc1394-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libopenexr-dev \
    openexr qv4l2 v4l-utils zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

# Set work directory in the container
WORKDIR /app

# Copy the local base-requirements.txt to the container
COPY /docker-base/base-requirements.txt /app/base-requirements.txt

# Install Python packages globally from base-requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r /app/base-requirements.txt && \
    pip install --no-cache-dir --prefer-binary h5py
