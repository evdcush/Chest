#!/bin/bash
# This script is a hack for an issue I've been experiencing on my system:
#   - Jammy
#   - 6.5.0-41-generic
#   - Xorg
#   - nvidia-drivers-550: `550.90.07`
#   - 2 x RTX 3090, with GPU0 having displays on HDMI-0 and HDMI-1
#
# for my current monitor setup:
#   - INNOCN 32M2V
#   - Dell G3223Q
#
# where, upon waking the displays after they have been turned-off by
# power management after a period of inactivity, the Dell monitor wakes without
# issue, but the INNOCN 32M2V monitor seems to wait for ~10s, during which time
# you cannot power-off the monitor, nor access its settings, then reports
# "no signal" from HDMI and will suspend.
#
# I've tried much, but ultimately, the only way to get the INNOCN monitor
# functional after wake is either to:
#   - wait the 10s until it reports no signal, so you can turn the monitor
#     on/off
# or
#   - wait another ~10-20s for the system to turn off the displays again
#     and *then*, on this second time, it will wake both monitors
#
# It's a monstrous waste of time.
# And I want to keep the power management settings, because they're both
# bright, 32" 4K monitors, so I don't want to waste the energy.
#
# So, I guess the only way is to have this HDMI ON/OFF hack script?
#
#
# !!!!!!!!!!!!!!!   IMPORTANT  !!!!!!!!!!!!!!!!
# This script is paired-with, and invoked by:
#   /etc/systemd/system/fix-monitor-display.service
# aka the `systemctl` service `fix-monitor-display.service`
#
xrandr --output HDMI-0 --off
xrandr --output HDMI-0 --auto
xrandr --output HDMI-1 --off
xrandr --output HDMI-1 --auto


# The content of the `fix-monitor-display.service`:
#
# [Unit]
# Description=Fix Monitor No-signal Wake Issue from Display Power Management
# After=display-manager.service
#
# [Service]
# User=evan
# Type=simple
# ExecStart=/home/evan/.local/bin/fix-monitor.sh
#
# [Install]
# WantedBy=multi-user.target
#
