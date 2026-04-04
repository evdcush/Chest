"""copy/clone/duplicate a Brave profile
brave deos not provide this functionality.
you ahve to do it live.

that wat this script for.
"""

import os
import shutil
import json
from pathlib import Path

# ge base path
#def get_user_data_path():
#    if os.name == 'nt':  # Windows
#        return home / "AppData/Local/BraveSoftware/Brave-Browser/User Data"
#    elif os.name == 'posix':
#        if (home / "Library").exists():  # macOS
#            return home / "Library/Application Support/BraveSoftware/Brave-Browser"
#        else:  # Linux
#            return home / ".config/BraveSoftware/Brave-Browser"
#    else:
#        raise Exception("Unsupported OS")

BRAVE_CONFIG_ROOT = '.config/BraveSoftware/Brave-Browser'


def get_linux_user_data_path():
    home = Path.home()
    dpath = home / BRAVE_CONFIG_ROOT
    return dpath

def copy_profile(src_dir, dst_dir):
    print('attempting copy profiel.')
    if dst_dir.exists():
        raise Exception(f"Destination '{dst_dir}' already exists.")
    shutil.copytree(src_dir, dst_dir)
    print('copied profile, probably.')

def update_local_state(user_data_path, dst_profile_dir, new_profile_name):
    local_state_file = user_data_path / "Local State"
    with open(local_state_file, "r+", encoding="utf-8") as f:
        state = json.load(f)
        info_cache = state.get("profile", {}).get("info_cache", {})
        info_cache[str(dst_profile_dir.name)] = {
            "name": new_profile_name
        }
        f.seek(0)
        json.dump(state, f, indent=2)
        f.truncate()

def clear_session_files(profile_path):
    print('clearing sess shit')
    for filename in ["Current Tabs", "Current Session", "Last Tabs", "Last Session", "Sessions"]:
        f = profile_path / filename
        if f.exists():
            print(f"found sess state file {filename}; unlnking")
            f.unlink()

def main():
    user_data_path = get_linux_user_data_path()
    print(f"Brave User Data Path: {user_data_path}")

    src_profile_name = input("Enter source profile name (e.g., Default, Profile 1): ")
    new_profile_name = input("Enter new profile name (e.g., Profile 5): ")
    new_display_name = input("Enter display name for the new profile (e.g., Chop): ")
    wipe_session = input("Clear tab/session data in new profile? (y/N): ").strip().lower() == "y"

    src_path = user_data_path / src_profile_name
    dst_path = user_data_path / new_profile_name

    if not src_path.exists():
        print(f"Source profile dir '{src_path}' not found.")
        return

    print(f"Copying '{src_path}' to '{dst_path}'...")
    copy_profile(src_path, dst_path)

    print("Updating 'Local State' with new profile info...")
    update_local_state(user_data_path, dst_path, new_display_name)

    if wipe_session:
        print("Clearing session files...")
        clear_session_files(dst_path)

    print(f"Profile '{new_display_name}' created as '{new_profile_name}'.")

if __name__ == "__main__":
    main()

