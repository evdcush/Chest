#!/usr/bin/bash

# INTENDED TO BE RAN WEEKLY!
#     `0 0 * * 0`

# SRC
src_fname='.zsh_history'
path_to_src_file="$HOME/$src_fname"

# DST
bkp_dst="$HOME/Documents/AutoBackups"
mkdir -p $bkp_dst


# formatting on the bkp filename
local_name=$(hostname)
bkp_date=$(date +"%F")

bkp_fname="zsh_history.$local_name.$bkp_date.txt"
path_to_dst_file="$bkp_dst/$bkp_fname"


# back that shit up
echo 'BACKUP:'
echo "    SRC:$path_to_src_file"
echo "    DST:$path_to_dst_file"

cp $path_to_src_file $path_to_dst_file

# eyeball it
echo 'Worked!(?)  CHECK:'
echo "$(ls -la $path_to_dst_file)"
