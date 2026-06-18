# Play with Reachy

Using hand pose estimation and action recognition to interact with an expressive Reachy Mini robot (AIxHRI Summer School 2026).

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

## Build and run

Build the image locally:

```bash
make build
```

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
