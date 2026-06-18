#!/usr/bin/env bash

set -euo pipefail

echo "========================================"
echo " Reachy Mini USB permissions setup"
echo "========================================"
echo
echo "This script requires sudo privileges."
echo "You may be prompted for your password."
echo

echo "Writing udev rules to /etc/udev/rules.d/99-reachy-mini.rules..."
cat <<'EOF' | sudo tee /etc/udev/rules.d/99-reachy-mini.rules >/dev/null
SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"
EOF

echo "Reloading udev rules and applying them..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Adding the current user to the dialout group..."
sudo usermod -aG dialout "$USER"

echo
echo "Done."
echo "You may need to log out and log back in for the group change to take effect."