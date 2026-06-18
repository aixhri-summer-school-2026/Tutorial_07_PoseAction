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

# Install Kyutai TTS and VLM dependencies
RUN uv pip install --python $VIRTUAL_ENV torch torchvision torchcodec "numpy>=2.0.0" "scipy>=1.14.0" soundfile huggingface_hub pocket-tts transformers Pillow ultralytics

# Install Silero VAD directly from PyPI and pre-cache the model
RUN uv pip install --python $VIRTUAL_ENV "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"
# RUN uv pip install --python $VIRTUAL_ENV "silero-vad" "faster-whisper"

# Pre-download the Pocket TTS model weights into the Docker image cache
RUN /opt/reachy_mini_env/bin/python -c "from pocket_tts import TTSModel; TTSModel.load_model(language='english')"

# # Pre-download the VLM weights (Safely downloads without loading into RAM)
# RUN /opt/reachy_mini_env/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openbmb/MiniCPM-V-4.6')"

# Install accelerate TODO: move this above
RUN uv pip install --python $VIRTUAL_ENV "accelerate"

# Pre-cache Silero VAD
# RUN /opt/reachy_mini_env/bin/python -c "from silero_vad import load_silero_vad; load_silero_vad()"

# Pre-cache Faster-Whisper 'tiny.en' model
# RUN /opt/reachy_mini_env/bin/python -c "from faster_whisper import download_model; download_model('tiny.en')"

# Add venv to PATH and set up auto-activation
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Add Faster-Whisper CUDA libraries to path
# ENV LD_LIBRARY_PATH="/opt/reachy_mini_env/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/reachy_mini_env/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

RUN echo "source /opt/reachy_mini_env/bin/activate" >> /home/reachy/.bashrc

# Set up color support for ls and grep
ENV TERM=xterm-256color
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /home/reachy/.bashrc && \
    echo "alias ls='ls --color=auto'" >> /home/reachy/.bashrc && \
    echo "alias grep='grep --color=auto'" >> /home/reachy/.bashrc

# Apply the camera detection patch
RUN sed -i 's/return find_video_device(devices)/return "\/dev\/reachy_camera", _make_camera_specs("Reachy")/' /opt/reachy_mini_env/lib/python3.12/site-packages/reachy_mini/media/device_detection.py

# Default command
CMD ["reachy-mini-tutorial-07-daemon"]