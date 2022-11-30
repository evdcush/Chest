# Android
Resources for my android devices.
All of which are currently samsung galaxy devices.

## Firmware

### `samloader`
https://github.com/samloader/samloader

> Download firmware for Samsung devices (without any extra Windows drivers).

Nice!

Install:

```sh
pip install git+https://github.com/samloader/samloader.git
```

Basic usage:

```sh
# Check the latest firmware version.
samloader -m SM-N9600 -r GTO checkupdate
# N9600ZHU9FVH2/N9600OWO9FVF3/N9600ZCU9FVH2/N9600ZHU9FVH2

```

Okay, nvm, I guess you can only download latest fm with samloader (same as frija).
I guess this is not an software limitation, but that samsung doesn't expose older versions maybe.
