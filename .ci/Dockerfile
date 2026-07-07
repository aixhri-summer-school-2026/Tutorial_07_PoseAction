FROM ubuntu:24.04

# Prevent tzdata and other interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
# Set up non-root user and group IDs with defaults that can be overridden at build time
ARG UID=1000
ARG GID=1000

# Update apt and install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    build-essential \
    pkg-config \
    libcamera-ipa \
    libcamera-tools \
    v4l-utils \
    libgstreamer-plugins-bad1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libglib2.0-dev \
    libssl-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    libportaudio2 \
    libnice10 \
    gstreamer1.0-tools \
    gstreamer1.0-libcamera \
    gstreamer1.0-plugins-good \
    gstreamer1.0-alsa \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-nice \
    gstreamer1.0-pulseaudio \
    pipewire \
    alsa-utils \
    python3-gi \
    python3-gi-cairo \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create the non-root user and add to dialout group for USB access
RUN groupadd -g ${GID} reachy
RUN useradd -m -u ${UID} -g reachy -s /bin/bash reachy
RUN usermod -aG dialout reachy
RUN usermod -aG audio reachy
RUN groupadd -f render && usermod -aG render reachy

# Initialize Git LFS
RUN git lfs install

# Create directories and give reachy ownership BEFORE switching users
RUN mkdir -p /opt/reachy_mini_env && chown reachy:reachy /opt/reachy_mini_env
RUN mkdir -p /opt/gst-plugins-rs && chown reachy:reachy /opt/gst-plugins-rs

# Switch to the non-root user
USER reachy

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/reachy/.cargo/bin:${PATH}"

# Build and install WebRTC plugin
WORKDIR /tmp
RUN git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git && \
    cd gst-plugins-rs && \
    git checkout 0.14.5 && \
    cargo install cargo-c && \
    cargo cinstall -p gst-plugin-webrtc --prefix=/opt/gst-plugins-rs --release && \
    rm -rf /tmp/gst-plugins-rs

# Set GStreamer plugin path
ENV GST_PLUGIN_PATH="/opt/gst-plugins-rs/lib/x86_64-linux-gnu:${GST_PLUGIN_PATH}"

# Return to app working directory
WORKDIR /app

# Install uv as reachy
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/reachy/.local/bin:${PATH}"

# Install Python 3.12 and create venv as reachy
RUN uv python install 3.12 --default
ENV VIRTUAL_ENV=/opt/reachy_mini_env
RUN uv venv $VIRTUAL_ENV --python 3.12

# Install reachy-mini directly into the venv
RUN uv pip install --python $VIRTUAL_ENV "reachy-mini"

# Install opencv
RUN uv pip install --python $VIRTUAL_ENV "opencv-python"

# Install dependencies
RUN uv pip install --python $VIRTUAL_ENV torch torchvision torchcodec "numpy>=2.0.0" "scipy>=1.14.0" soundfile huggingface_hub pocket-tts transformers Pillow ultralytics

# Install NVIDIA CUDA libraries and onnxruntime-gpu for GPU inference (see tutorial/update.sh)
RUN uv pip install --python $VIRTUAL_ENV \
    "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*" \
    "onnxruntime-gpu==1.19.2" \
    nvidia-cuda-runtime-cu12 \
    nvidia-curand-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cuda-nvrtc-cu12
# RUN uv pip install --python $VIRTUAL_ENV "silero-vad" "faster-whisper"

# Install accelerate TODO: move this above
RUN uv pip install --python $VIRTUAL_ENV "accelerate"

# Pre-download RTMLib whole-body models (see visualize_wholebody_live.py)
USER root
RUN apt-get update && apt-get install -y unzip
RUN mkdir -p /app/downloads && \
    wget -q -P /app/downloads \
      "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip" \
      "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip" && \
    unzip -q /app/downloads/yolox_m_8xb8-300e_humanart-c2c7a14a.zip -d /app/downloads/yolox_det && \
    unzip -q /app/downloads/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip -d /app/downloads/rtmw_pose && \
    mv "$(find /app/downloads/yolox_det -name end2end.onnx)" /app/downloads/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx && \
    mv "$(find /app/downloads/rtmw_pose -name end2end.onnx)" /app/downloads/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.onnx && \
    rm -rf /app/downloads/*.zip /app/downloads/yolox_det /app/downloads/rtmw_pose && \
    chown -R reachy:reachy /app/downloads
USER reachy

# Add venv to PATH and set up auto-activation
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"


# Register NVIDIA libraries system-wide and mirror tutorial/update.sh for interactive shells
USER root
RUN /opt/reachy_mini_env/bin/python -c "\
import glob; \
paths = glob.glob('/opt/reachy_mini_env/lib/python3.12/site-packages/nvidia/*/lib'); \
open('/etc/ld.so.conf.d/reachy-nvidia.conf', 'w').write('\n'.join(paths) + '\n')" && \
    ldconfig
USER reachy
RUN NVIDIA_LIBS=$(/opt/reachy_mini_env/bin/python -c "import glob; print(':'.join(glob.glob('/opt/reachy_mini_env/lib/python3.12/site-packages/nvidia/*/lib')))") && \
    echo "export LD_LIBRARY_PATH=\"${NVIDIA_LIBS}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" >> /home/reachy/.bashrc

RUN echo "source /opt/reachy_mini_env/bin/activate" >> /home/reachy/.bashrc

# Set up color support for ls and grep
ENV TERM=xterm-256color
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /home/reachy/.bashrc && \
    echo "alias ls='ls --color=auto'" >> /home/reachy/.bashrc && \
    echo "alias grep='grep --color=auto'" >> /home/reachy/.bashrc

# Apply the camera detection patch
RUN sed -i 's/return find_video_device(devices)/return "\/dev\/reachy_camera", _make_camera_specs("Reachy")/' /opt/reachy_mini_env/lib/python3.12/site-packages/reachy_mini/media/device_detection.py

# Default command
CMD ["reachy-mini-daemon"]