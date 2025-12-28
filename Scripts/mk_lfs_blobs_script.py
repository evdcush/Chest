#!/usr/bin/env python
# copyright evdcush 2025
"""
simple (elaborate, probably unnecessary if not grug mind)
script to make a shell script to acquire lfs targets


pull script should look like::

    #!/usr/bin/env bash

    echo 'downloading: 64-8bits.tflite'; git-lfs pull --include='64-8bits.tflite' && rm -rf .git/lfs/objects; echo '';
    echo 'downloading: 64-fp16.tflite'; git-lfs pull --include='64-fp16.tflite' && rm -rf .git/lfs/objects; echo '';
    echo 'downloading: 64.tflite'; git-lfs pull --include='64.tflite' && rm -rf .git/lfs/objects; echo '';


output should look like::

    downloading: 64-8bits.tflite
    Downloading LFS objects: 100% (1/1), 125 MB | 69 MB/s
    downloading: 64-fp16.tflite
    Downloading LFS objects: 100% (1/1), 248 MB | 420 MB/s
    downloading: 64.tflite
    Downloading LFS objects: 100% (1/1), 496 MB | 8008135 MB/s
"""

import subprocess

# init the sh file string
dst_filename = 'pull_lfs_blobs.sh'
file_str = '#!/usr/bin/env bash\n\n'
cmd_str  = (
    "echo 'downloading: {blob}'; "
    "git-lfs pull --include='{blob}' && "
    "rm -rf .git/lfs/objects; "
    "echo '';"
)


# file permissions
# ================
# NB: `os.chmod` mode arg MUST be octal rep (like actual posix).
#     so int `754` != permissions 754
#file_permissions_mode = 0o754
# STOP going turbo on this; just fcking use subprocess, fucl..

# get the LFS targets
# ===================
lfs_targets_ret = subprocess.run(
    'git-lfs ls-files -n',
    shell=True,
    capture_output=True,
    encoding='utf-8',
)
# looks somethin like:
'''
CompletedProcess(
    args='git-lfs ls-files -n',
    returncode=0,
    stdout='flax_model.msgpack\nmodel.safetensors\npytorch_model.bin',
    stderr=''
)
'''

# process the output
# ------------------
lfs_targets_str = lfs_targets_ret.stdout.strip()
# looks somethin like:
#     'flax_model.msgpack\nmodel.safetensors\npytorch_model.bin'
assert lfs_targets_str, f"NO LFS FILES FOUND! sub run ret: {lfs_targets_ret}"

lfs_targets_lst = lfs_targets_str.split('\n')

# Make the script lines
# =====================
script_cmds_lst = []

for lfs_target in lfs_targets_lst:
    # (lfs filename) --> (shell cmd line), eg:
    #     lfs_target:  'flax_model.msgpack'
    #     fmt_lfs_target:
    # "git-lfs pull --include='flax_model.msgpack' && rm -rf .git/lfs/objects;"
    fmt_lfs_target = cmd_str.format(blob=lfs_target)
    print(f"LFS TARGET: {lfs_target}\n   GET CMD: {fmt_lfs_target}")
    script_cmds_lst.append(fmt_lfs_target)

# get the final cmds string
script_cmds = '\n'.join(script_cmds_lst)
# get the final file string
complete_file_str = file_str + script_cmds

# Make the shell script
# =====================
with open(dst_filename, 'x') as getter_file:
    getter_file.write(complete_file_str)

# make it executable
subprocess.run(
    f"chmod +x {dst_filename}",
    shell=True,
)

# fucking done

