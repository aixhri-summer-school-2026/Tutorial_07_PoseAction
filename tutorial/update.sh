#!/bin/bash
uv pip uninstall onnxruntime-gpu --python /opt/reachy_mini_env
uv pip install "onnxruntime-gpu==1.19.2" nvidia-cuda-runtime-cu12 --python /opt/reachy_mini_env
uv pip install nvidia-cuda-runtime-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12 --python /opt/reachy_mini_env


LIBS=$(python -c "
import glob
print(':'.join(glob.glob('/opt/reachy_mini_env/lib/python3.12/site-packages/nvidia/*/lib')))
")

echo "LIBS=$LIBS"

export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
