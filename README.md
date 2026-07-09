# Play with Reachy

Using whole-body pose estimation and gesture recognition to interact with an expressive Reachy Mini robot (AIxHRI Summer School 2026).

You will find here all the install instructions. If you already have the docker use `make install-rules`, `make run` then `make shell`.

> [!TIP]
> Go to the [tutorial instructions](tutorial/README.md).

## Prerequisites

- Docker and Docker Compose installed (see [docker setup guide](https://github.com/aixhri-summer-school-2026/docker-nvidia-tuto)).
- Reachy Mini connected by USB.
- Linux host with permission to run `sudo`.

## Setup

Clone the repository:

```bash
git clone git@github.com:aixhri-summer-school-2026/Tutorial_07_PoseAction.git
cd Tutorial_07_PoseAction
```

Install udev rules (USB + camera symlink):

```bash
make install-rules
```

If group permissions were updated, log out and log back in once.

## Build the docker (if you have not pulled it)

Build the image locally:

```bash
make build
```

## Run the docker
Start the Reachy Mini daemon container:

```bash
make run
```

Open a shell inside the running container:

```bash
make shell
```

Stream logs:

```bash
make logs
```

Stop everything:

```bash
make down
```

## Tests

Test scripts are in `tests/` and can be run from inside the container.

See `tests/README.md` for full test instructions.

## Troubleshooting

- If you pulled or built the docker **before 08/07/2026**, the whole-body ONNX models are not baked into the image. Either rebuild (`make build`) or do the following (faster):

```bash
# 1. On the host (outside the docker): download and extract the models into tutorial/
mkdir -p tutorial
cd tutorial
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip
wget https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip
unzip yolox_m_8xb8-300e_humanart-c2c7a14a.zip -d yolox_det
unzip rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip -d rtmw_pose
mv "$(find yolox_det -name end2end.onnx)" yolox_m_8xb8-300e_humanart-c2c7a14a.onnx
mv "$(find rtmw_pose -name end2end.onnx)" rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.onnx
rm -rf *.zip yolox_det rtmw_pose
cd ..

# 2. Inside the docker: move them to /app/downloads/ (where the live scripts look)
make shell
mkdir -p /app/downloads
mv /app/tutorial/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx /app/downloads/
mv /app/tutorial/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.onnx /app/downloads/

# 3. Install onnxruntime-gpu and fix LD_LIBRARY_PATH (everytime you do a new make down / make run)
source tutorial/update.sh
```

Note: `/app/downloads/` is not bind-mounted, so if you recreate the container you must repeat step 2 (the `.onnx` files stay on the host in `tutorial/`).


- If you have issues with "permission denied" for video devices (integrated or usb webcam). Outside of the docker, do the following :
```
ls -la /dev/video*
sudo chmod 666 /dev/video2   # replace with your desired cam device
```

- The volume is too low or too high (70 to 80 recommended)
```
amixer -c 0 sset 'PCM' 75%
amixer -c 0 sset 'PCM',1 75%
```

- If you have display issues like : `qt.qpa.xcb: could not connect to display :1`
```
# On the host (outside of the docker)
xhost +local:docker
```