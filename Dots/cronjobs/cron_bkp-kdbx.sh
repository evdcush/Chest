#!/usr/bin/bash

# INTENDED TO BE RAN WEEKLY!
#     `0 0 * * 0`

# SRC
src_fname='kagi_da_yo.kdbx'
path_to_src_file="$HOME/Documents/Cloud/Document/$src_fname"

# DST
bkp_dst="$HOME/Documents/AutoBackups"
mkdir -p $bkp_dst


# formatting on the bkp filename
filename=$(basename $path_to_src_file)  # 'kagi_da_yo.kdbx'
name="${filename%.*}"                   # 'kagi_da_yo'
ext="${filename##*.}"                   # 'kdbx'

# bkp filename
bkp_date=$(date +"%F")                  # YYYY-MM-DD
bkp_fname="$name.$bkp_date.$ext"        # 'kagi_da_yo.1969-04-20.kdbx'
path_to_dst_file="$bkp_dst/$bkp_fname"


# back that shit up
echo 'BACKUP:'
echo "    SRC:$path_to_src_file"
echo "    DST:$path_to_dst_file"

cp $path_to_src_file $path_to_dst_file

# eyeball it
echo 'Worked!(?)  CHECK:'
echo "$(ls -la $path_to_dst_file)"
