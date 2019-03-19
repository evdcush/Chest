#!/usr/bin/env python
"""
wrapper for nativefier

Primary features of this wrapper:
- automatic creation of desktop file
- building apps to one location and linking
- managing icon source

Shortcomings:
- assumed that you have downloaded a png icon asset
    - nativefier's automatic sourcing of icons is hit-or-miss

"""
import os
import sys
import code
import subprocess
import utils

# Parser
cli = utils.CLI

# Nativefier args
_DEFAULT_ARGS = '--file-download-options \'{"saveAs": true}\' -m' # saveto dialog, menu bar

cli.add_argument('url', type=str,
    help='url pointing to target web app')

cli.add_argument('name', type=str,
    help='name of nativefied app')

#cli.add_argument('-i', '--icon', type=str, help='.png icon for app')

cli.add_argument('-u', '--internal_urls', type=str,
    help='regex of urls to be considered in domain, eg ".*?\\.google\\.*?"')

cli.add_argument('-t', '--tray', action='store_true',
    help='whether app remains in tray when closed')

cli.add_argument('-m', '--multi-instance', action='store_true',
    help='allow multiple instances of app')

cli.add_argument('-c', '--counter', action='store_true',
    help='counter num for window label (for supporting apps)')


# Paths
bin_path = utils.BIN
app_path = utils.NATIVEFIED
icons_path   = utils.ICONS
desktop_path = utils.DESKTOPS

# Desktop file
DESKTOP_ENTRY = f"""\
[Desktop Entry]
Name={{title}}
Comment=Nativefied {{title}}
Exec={bin_path}/{{name}}
Terminal=false
Type=Application
Icon={app_path}/{{name}}-linux-x64/resources/app/icon.png
Categories=Network;
"""

def write_desktop_file(name):
    title = name.title()
    desktop_entry = DESKTOP_ENTRY.format(title=title, name=name)
    with open(desktop_path + f'/{name}.desktop', 'w') as dfile:
        dfile.write(desktop_entry)
    print(f"Wrote desktop entry:\n{desktop_entry}")

def mkdirs(p):
    if not os.path.exists(p):
        os.makedirs(p)

def main():
    # create dirs if missing
    mkdirs(app_path)
    mkdirs(bin_path)

    # Parse args
    args = cli.parse_args()
    url  = args.url
    name = args.name
    path = app_path + f'/{name}-linux-x64'
    local_bin = f'{path}/{name}'

    # Check icon exists (and properly named)
    icon = f'{icons_path}/{name}.png'
    if not os.path.exists(icon):
        msg  = f'\n\tEXPECTED: {icon}'
        msg += f'\n\tMust download a valid png icon asset prior to nativefying'
        raise FileNotFoundError(msg)

    # Interpret nativefier command
    cmd = f"nativefier {url} -n {name} -i {icon} "
    if not args.multi_instance:
        cmd += '--single-instance '
    if args.tray:
        cmd += '--tray '
    if args.counter:
        cmd += '--counter '
    if args.internal_urls:
        cmd += f'--internal-urls {args.internal_urls} '
    cmd += _DEFAULT_ARGS

    print(f"NATIVEFIER COMMAND:\n{cmd}\n\n")
    #code.interact(local=dict(globals(), **locals()))

    # Run command and symlink
    subprocess.run(cmd, cwd=app_path, shell=True)
    subprocess.run(f'chmod +x {local_bin}', shell=True)
    subprocess.run(f'ln -sf {local_bin} {bin_path}', shell=True)

    # create desktop file
    write_desktop_file(name)
    return 0


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)

