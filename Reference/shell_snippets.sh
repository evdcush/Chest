#!/bin/bash

#=============================================================================#
#                                                                             #
#                             _          _                     __             #
#                            (_)        | |                   / _|            #
#              __ _   _   _   _    ___  | | __  _ __    ___  | |_             #
#             / _` | | | | | | |  / __| | |/ / | '__|  / _ \ |  _|            #
#            | (_| | | |_| | | | | (__  |   <  | |    |  __/ | |              #
#             \__, |  \__,_| |_|  \___| |_|\_\ |_|     \___| |_|              #
#                | |                                                          #
#                |_|                                                          #
#                                                                             #
#=============================================================================#

# read data from user
#----------------------------
echo "Enter a value: "
read userInput
echo "You just entered $userInput"


# Write to file
#----------------------------
#==== over-write file
echo "Hello world" > 'foo.txt'

#==== concatenate to end file instead
echo "Hello world" >> 'foo.txt'

# Open most recent command and edit it
# ------------------------------------
fc
fc -e 'vim' # specify editor
fc -l # list recent cmds from history


# Sort file
#----------------------------
sort input-file > output_file
#==== sort in-place
sort -o file file


# Get set intersection/difference
# between to text files
#----------------------------
# use `comm`
#==== vanilla comm
# tab-separated cols:
# [lines unique to F1] [lines unique to F2] [lines in common]
comm file1 file2

#==== only common to both
comm -12 file1 file2
cat file1 | comm -12 - file2  # where file1 is read from STDIN

#==== lines only found in F1
comm -23 file1 file2

#==== only lines found in F2, when files arent sorted
comm -13 <(sort file1) <(sort file2)


# Get unique lines in file (no dups)
#-----------------------------
#==== de-duplicated, single lines
sort file | uniq
sort file | uniq -i  # ignore case

#==== unique and duplicate
sort file | uniq -u  # unique lines only
sort file | uniq -d  # duplicates only

# Get name of machine
# -------------------
# `hostnamectl` and `cat /proc/sys/kernel/hostname`, also good
hostname
#-----> T4


# mv stuff
# -------------------
#====== Rename file, without retyping dir path:
mv very/long/path/to/filename.{old,new}
#----  vs
mv very/long/path/to/filename.old very/long/path/to/filename.new

#==== move multiple things
# of course, typical wildcard match
mv *.ipynb notebook_dir/
# but for non-matching...
mv -t DESTINATION file1 file2 file3 ...
mv file1 file2 file3 -t DESTINATION


# Manipulating Strings
# --------------------
str=https://github.com/willsALMANJ/Zutilo
# CUT
#     1      2        3             4           5
# ['https:', '', 'github.com', 'willALMANJ', 'Zutilo']
echo "$str" | cut -d'/' -f5 # ---> Zutilo
echo "$str" | cut -d'/' -f4 # ---> wukksALMANJ

# tr (translate or delete chars)
echo $str | tr "/" "\n" | tail -n 2
# willsALMANJ
# Zutilo

# Adjust/view timezone
# --------------------
# Check current timezone settings
timedatectl

# list timezones
timedatectl list-timezones
# eg:
timedatectl list-timezones | grep -i america
timedatectl list-timezones | fzf

# Change timezone
sudo timedatectl set-timezone Asia/Tokyo # painless

# painful way
sudo unlink /etc/localtime
sudo ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime




#=============================================================================#
#                                                                             #
#                            _                          _                     #
#                           (_)                        | |                    #
#              ___   _ __    _   _ __    _ __     ___  | |_   ___             #
#             / __| | '_ \  | | | '_ \  | '_ \   / _ \ | __| / __|            #
#             \__ \ | | | | | | | |_) | | |_) | |  __/ | |_  \__ \            #
#             |___/ |_| |_| |_| | .__/  | .__/   \___|  \__| |___/            #
#                               | |     | |                                   #
#                               |_|     |_|                                   #
#                                                                             #
#=============================================================================#

# Get list of orphaned desktop entries
#======================================
for i in {/usr,~/.local}/share/applications/*.desktop; do which $(grep -Poh '(?<=Exec=).*?( |$)' $i) > /dev/null || echo $i; done

# Remove all items from output of `find`
#======================================
# Find and delete: CAREFUL
function find_and_remove_all(){
    matches=$(sudo find / -iname "*$1*")
    echo "$matches"
    echo -n "Do you want to proceed (y/n)? "
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ;then
        echo ""
        sudo find / -iname "*$1*" -exec rm -rf "{}" \;
    else
        echo Aborted
    fi
}

# Fix broken symlinks
# ===================
find /mnt/home/someone/something -lname '/home/someone/*' \
   # xec sh -c 'ln -snf "/mnt$(readlink "$0")" "$0"' {} \;

# don't remember what this went to, but keeping it because it must have
# felt like a cmd worth saving
#sed 's/.*private.*$//ig' m.txt


# Get latest release from GitHub API
# ==================================
curl --silent "https://api.github.com/repos/USER/REPO/releases/latest" |
    grep '"tag_name":' |
    sed -E 's/.*"([^"]+)".*/\1/' |
    xargs -I {} curl -sOL "https://github.com/USER/REPO/archive/"{}'.tar.gz'

# Or, using JQ
curl https://api.github.com/repos/willsALMANJ/Zutilo/releases/latest | jq '.assets | .[0] | .browser_download_url'
# "https://github.com/willsALMANJ/Zutilo/releases/download/v3.0.3/zutilo.xpi"
curl https://api.github.com/repos/willsALMANJ/Zutilo/releases/latest \
| jq '.assets | .[0] | .browser_download_url' | xargs wget



# Python embedded in sh function
# ==============================
function hurl {
ARGS="$@" IPATH="$PATH_INBOX_HOARD" python - <<END
import os
import yaml

#==== Read STDIN
path_inbox = str(os.environ['IPATH'])
args_input = os.environ['ARGS'].split(' ')

#==== Process args
has_flag  = len(args_input) > 1
flag, url = args_input if has_flag else ('r', args_input.pop())

#==== Get key
key_map = dict(r="repos", o="orgs", u="users")
key = key_map[flag]

#==== Get inbox file
with open(path_inbox) as file_inbox:
    inbox = yaml.load(file_inbox)

#==== Update inbox
# > NB: duplicate entries managed by hoarding script
inbox[key].append(url)
with open(path_inbox, 'w') as file_inbox:
    yaml.dump(inbox, file_inbox, default_flow_style=False)
print(f'SAVED: inbox[{key}].append({url})')
END
}


# MISC/deprec. cmds useful for reference
# ======================================

# Reading GH auth tokens from structured plain text (yaml)
#=== pure sh utils only:
TOKEN_GH_SCRAPE="$(cat $GH_TOKENS \
| grep token \
| awk '{print $2}' \
| tail -1)"

#=== utilizing yq:
TOKEN_GH_CRX="$(yq r $GH_TOKENS public.crx.token)"
TOKEN_GH_SCRAPE="$(yq r $GH_TOKENS public.scrape.token)"

#=== "ternary" type check on stdin args
deprec_getstars(){ # DEPRECATED; just made my own script an executable
    # if user specified, get their stars, else get mine
    USER=${1:-'evdcush'}   # slickkkkkkkkkkkk, love this usage of param exp.
    TOKEN="$TOKEN_GH_SCRAPE"
    getstars -u "$USER" -t "$TOKEN" -s > "GH-Stars_$USER.md"
}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ============= #
# miscellaneous #
# ============= #

# Get chrome extension file
#    TODO: function not properly tested. And it's ugly.
get_crx(){
    CRXNAME="$1.crx"
    CRXID="$2"
    CRX_PATH="$DIR_MEDIA/Software/CRX"
    A2="crx?response=redirect&acceptformat=crx2,crx3&prodversion=69"
    A3="&x=id%3D$CRXID%26installsource%3Dondemand%26uc"
    echo $CRX_PATH
    wget -O "$CRXNAME" "https://clients2.google.com/service/update2/$A2$A3"
}

# DEPRECATED: `unar` is used instead
function extract() {
    #==== Extract many types of compressed packages
    if [ -f "$1" ] ; then
        case "$1" in
            *.tar.bz2)   tar xvjf "$1"                    ;;
            *.tar.gz)    tar xvzf "$1"                    ;;
            *.bz2)       bunzip2 "$1"                     ;;
            *.rar)       unrar x "$1"                     ;;
            *.gz)        gunzip "$1"                      ;;
            *.tar)       tar xvf "$1"                     ;;
            *.tbz2)      tar xvjf "$1"                    ;;
            *.tgz)       tar xvzf "$1"                    ;;
            *.zip)       unzip "$1"                       ;;
            *.ZIP)       unzip "$1"                       ;;
            *.pax)       cat "$1" | pax -r                ;;
            *.pax.Z)     uncompress "$1" —stdout | pax -r ;;
            *.Z)         uncompress "$1"                  ;;
            *.7z)        7z x "$1"                        ;;
            *)           echo "don't know how to extract '$1'..." ;;
        esac
    else
        echo "extract: error: $1 is not valid"
    fi
}

#----- Add papers/literature to reading inbox
function rurl {
    #TARGET='arxiv'
    #URL="$1"
    #if [ "$#" -gt 1 ]; then
    #    #==== 'other'
    #    TARGET="$1"
    #    URL="$2"
    #fi
    #yq w -i $PATH_INBOX_READ "$TARGET"'[+]' $URL
    echo $1 >> "$HOME/Cloud/Reading/inbox.txt"
}


# Add apt repo
# ============
# DEPRECATED: more of a PITA to clip just part of command than just whole cmd
#addrep(){
#    sudo add-apt-repository "ppa:$1" -y
#    sudo apt-fast update
#}

#---- git repo hoarding (cloning)
function hurl {
    TARGET='repos'
    URL="$1"
    if [ "$#" -gt 1 ]; then
        #==== "orgs" or "users" specified
        TARGET="$1"
        URL="$2"
    fi
    yq w -i $PATH_INBOX_HOARD "$TARGET"'[+]' $URL
}


# Lorem ipsum
# ===========
# DEPRECATED: python faker
#li() { lorem-ipsum $@ | xclip -selection clipboard }



# Get bib info
# ============
# DEPRECATED: using my own api now
#function arxbib {
#    arxiv2bib $1 | xclip -selection clipboard
#}


##---- add line to shell config
# DEPRECATED: almost never used
#function expzsh(){
#    echo "\n\n#==== $1" >> ~/.zshrc
#    echo "$2" >> ~/.zshrc
#}


# Timezone diff
# =============
function jst2pst(){
    # *** ASSUMES SYSTEM tz IS PST ***
    _TIME="$1"
    _DATE="$2"
    JST='TZ="Asia/Tokyo"'
    # Sample call:
    # date --date='TZ="Asia/Tokyo" 09:00 2/1'
    date --date="$JST $_TIME $_DATE"
}


#https://api.github.com/repos/willsALMANJ/Zutilo/releases/latest
# Get latest release
# ==================
# NB: this assumes whatever target asset you want is
#     the first asset listed in release
function gh-release(){
    if [ "$#" -eq 0 ]; then
        args=`xclip -o -selection clipboard` # NB: wrapping bash statement in backticks makes var = statement ret
    else
        args=$1
    fi
    splitargs=`echo $args | tr "/" "\n" | tail -n 2`
    read -d "\n" GH_USER GH_REPO <<<$splitargs
    #echo $GH_USER
    #echo $GH_REPO
    api_url="https://api.github.com/repos/$GH_USER/$GH_REPO/releases/latest"
    curl $api_url | jq '.assets | .[0] | .browser_download_url' | xargs wget
}

#=============================================================================#
#                                                                             #
#                 .d8888b.   .d88888b.  8888888b.  8888888888                 #
#                d88P  Y88b d88P" "Y88b 888   Y88b 888                        #
#                888    888 888     888 888    888 888                        #
#                888        888     888 888   d88P 8888888                    #
#                888        888     888 8888888P"  888                        #
#                888    888 888     888 888 T88b   888                        #
#                Y88b  d88P Y88b. .d88P 888  T88b  888                        #
#                 "Y8888P"   "Y88888P"  888   T88b 8888888888                 #
#                                                                             #
#                                                                             #
#             888     888 88888888888 8888888 888       .d8888b.              #
#             888     888     888       888   888      d88P  Y88b             #
#             888     888     888       888   888      Y88b.                  #
#             888     888     888       888   888       "Y888b.               #
#             888     888     888       888   888          "Y88b.             #
#             888     888     888       888   888            "888             #
#             Y88b. .d88P     888       888   888      Y88b  d88P             #
#              "Y88888P"      888     8888888 88888888  "Y8888P"              #
#                                                                             #
#=============================================================================#


#=============================================================================#
#     _____   _   _            ____                  _                        #
#    |  ___| (_) | |   ___    / ___|   _   _   ___  | |_    ___   _ __ ___    #
#    | |_    | | | |  / _ \   \___ \  | | | | / __| | __|  / _ \ | '_ ` _ \   #
#    |  _|   | | | | |  __/    ___) | | |_| | \__ \ | |_  |  __/ | | | | | |  #
#    |_|     |_| |_|  \___|   |____/   \__, | |___/  \__|  \___| |_| |_| |_|  #
#                                      |___/                                  #
#                                                                             #
#=============================================================================#

#-----------------------------------------------------------------------------#
#                                     cat                                     #
#-----------------------------------------------------------------------------#
# Display the contents of a file
cat /path/to/foo

# Display contents with line numbers
cat -n /path/to/foo

# concat several files into the target file:
cat file1 file2 > target_file # >> will append

# Display contents with line numbers (blank lines excluded)
cat -b /path/to/foo


#-----------------------------------------------------------------------------#
#                                    chmod                                    #
#-----------------------------------------------------------------------------#
# Add execute for all (myscript.sh)
chmod a+x myscript.sh

# Set user to read/write/execute, group/global to read only (myscript.sh), symbolic mode
chmod u=rwx, go=r myscript.sh

# Remove write from user/group/global (myscript.sh), symbolic mode
chmod a-w myscript.sh

# Remove read/write/execute from user/group/global (myscript.sh), symbolic mode
chmod = myscript.sh

# Set user to read/write and group/global read (myscript.sh), octal notation
chmod 644 myscript.sh

# Set user to read/write/execute and group/global read/execute (myscript.sh), octal notation
chmod 755 myscript.sh

# Set user/group/global to read/write (myscript.sh), octal notation
chmod 666 myscript.sh

# # Roles
# u - user (owner of the file)
# g - group (members of file's group)
# o - global (all users who are not owner and not part of group)
# a - all (all 3 roles above)
#
# # Numeric representations
# 7 - full (rwx)
# 6 - read and write (rw-)
# 5 - read and execute (r-x)
# 4 - read only (r--)
# 3 - write and execute (-wx)
# 2 - write only (-w-)
# 1 - execute only (--x)
# 0 - none (---)

#-----------------------------------------------------------------------------#
#                                    chown                                    #
#-----------------------------------------------------------------------------#
# Change file owner
chown user file

# Change file owner and group
chown user:group file

# Change owner recursively
chown -R user directory

# Change ownership to match another file
chown --reference=/path/to/ref_file file

#-----------------------------------------------------------------------------#
#                                     cmp                                     #
#-----------------------------------------------------------------------------#
# Compare two files

# Find the byte number and line number of the first difference between the files:
cmp file1 file2

# Find the byte number and differing bytes of every difference:
cmp -l file1 file2


#-----------------------------------------------------------------------------#
#                                    file                                     #
#-----------------------------------------------------------------------------#
# Determine file type.

# Give a description of the type of the specified file. Works fine for files with no file extension:
file filename

# Look inside a zipped file and determine the file type(s) inside:
file -z foo.zip

# Allow file to work with special or device files:
file -s filename

# Don't stop at first file type match; keep going until the end of the file:
file -k filename

# Determine the mime encoding type of a file:
file -i filename


#-----------------------------------------------------------------------------#
#                                     tee                                     #
#-----------------------------------------------------------------------------#
# Read from standard input and write to standard output and files (or commands).

# Copy standard input to each FILE, and also to standard output:
echo "example" | tee FILE

# Append to the given FILEs, do not overwrite:
echo "example" | tee -a FILE

# Print standard input to the terminal, and also pipe it into another program for further processing:
echo "example" | tee /dev/tty | xargs printf "[%s]"

# Create a folder called "example", count the number of characters in "example" and write "example" to the terminal:
echo "example" | tee >(xargs mkdir) >(wc -c)


#-----------------------------------------------------------------------------#
#                                    split                                    #
#-----------------------------------------------------------------------------#
# Split a file into pieces.

# Split a file, each split having 10 lines (except the last split):
split -l 10 filename

# Split a file into 5 files. File is split such that each split has same size (except the last split):
split -n 5 filename

# Split a file with 512 bytes in each split (except the last split; use 512k for kilobytes and 512m for megabytes):
split -b 512 filename

# Split a file with at most 512 bytes in each split without breaking lines:
split -C 512 filename


#=============================================================================#
#                           _____                 _                           #
#                          |_   _|   ___  __  __ | |_                         #
#                            | |    / _ \ \ \/ / | __|                        #
#                            | |   |  __/  >  <  | |_                         #
#                            |_|    \___| /_/\_\  \__|                        #
#                                                                             #
#=============================================================================#

#-----------------------------------------------------------------------------#
#                                     awk                                     #
#-----------------------------------------------------------------------------#
# A versatile programming language for working on files.

# Print the fifth column (a.k.a. field) in a space-separated file:
awk '{print $5}' filename

# Print the second column of the lines containing "something" in a space-separated file:
awk '/something/ {print $2}' filename

# Print the last column of each line in a file, using a comma (instead of space) as a field separator:
awk -F ',' '{print $NF}' filename

# Sum the values in the first column of a file and print the total:
awk '{s+=$1} END {print s}' filename

# Sum the values in the first column and pretty-print the values and then the total:
awk '{s+=$1; print $1} END {print "--------"; print s}' filename

# Print every third line starting from the first line:
awk 'NR%3==1' filename

# sum integers from a file or stdin, one integer per line:
printf '1\n2\n3\n' | awk '{ sum += $1} END {print sum}'

# using specific character as separator to sum integers from a file or stdin
printf '1:2:3' | awk -F ":" '{print $1+$2+$3}'

# print a multiplication table
seq 9 | sed 'H;g' | awk -v RS='' '{for(i=1;i<=NF;i++)printf("%dx%d=%d%s", i, NR, i*NR, i==NR?"\n":"\t")}'

# Specify output separator character
printf '1 2 3' | awk 'BEGIN {OFS=":"}; {print $1,$2,$3}'


#-----------------------------------------------------------------------------#
#                                  basename                                   #
#-----------------------------------------------------------------------------#
# Returns non-directory portion of a pathname.

# Show only the file name from a path:
basename path/to/file
# basename Projects/dummy_proj/src/hello.cpp ---> hello.cpp

# Show only the file name from a path, with a suffix removed:
basename path/to/file suffix


#-----------------------------------------------------------------------------#
#                                    comm                                     #
#-----------------------------------------------------------------------------#
# Select or reject lines common to two files. Both files must be sorted.

# Produce three tab-separated columns: lines only in first file, lines only in second file and common lines:
comm file1 file2

# Print only lines common to both files:
comm -12 file1 file2

# Print only lines common to both files, reading one file from stdin:
cat file1 | comm -12 - file2

# Get lines only found in first file, saving the result to a third file:
comm -23 file1 file2 > file1_only

# Print lines only found in second file, when the files aren't sorted:
comm -13 <(sort file1) <(sort file2)

#-----------------------------------------------------------------------------#
#                                   csplit                                    #
#-----------------------------------------------------------------------------#
# Split a file into pieces.
# This generates files named "xx00", "xx01", and so on.

# Split a file at lines 5 and 23:
csplit file 5 23

# Split a file every 5 lines (this will fail if the total number of lines is not divisible by 5):
csplit file 5 {*}

# Split a file every 5 lines, ignoring exact-division error:
csplit -k file 5 {*}

# Split a file at line 5 and use a custom prefix for the output files:
csplit file 5 -f prefix

# Split a file at a line matching a regular expression:
csplit file /regex/




#-----------------------------------------------------------------------------#
#                                     cut                                     #
#-----------------------------------------------------------------------------#
# Cut out fields from STDIN or files.

# Cut out the first sixteen characters of each line of STDIN:
cut -c 1-16

# Cut out the first sixteen characters of each line of the given files:
cut -c 1-16 file

# Cut out everything from the 3rd character to the end of each line:
cut -c 3-

# Cut out the fifth field of each line, using a colon as a field delimiter (default delimiter is tab):
cut -d':' -f5

# Cut out the 2nd and 10th fields of each line, using a semicolon as a delimiter:
cut -d';' -f2,10

# Cut out the fields 3 through to the end of each line, using a space as a delimiter:
cut -d' ' -f3-

#-----------------------------------------------------------------------------#
#                                     sed                                     #
#-----------------------------------------------------------------------------#
# Edit text in a scriptable manner.

# Replace the first occurrence of a regular expression in each line of a file, and print the result:
sed 's/regex/replace/' filename

# Replace all occurrences of an extended regular expression in a file, and print the result:
sed -r 's/regex/replace/g' filename

# Replace all occurrences of a string in a file, overwriting the file (i.e. in-place):
sed -i 's/find/replace/g' filename

# Replace only on lines matching the line pattern:
sed '/line_pattern/s/find/replace/' filename

# Delete lines matching the line pattern:
sed '/line_pattern/d' filename

# Print only text between n-th line till the next empty line:
sed -n 'line_number,/^$/p' filename

# Apply multiple find-replace expressions to a file:
sed -e 's/find/replace/' -e 's/find/replace/' filename

# Replace separator / by any other character not used in the find or replace patterns, e.g., #:
sed 's#find#replace#' filename


#-----------------------------------------------------------------------------#
#                                    tail                                     #
#-----------------------------------------------------------------------------#
# Display the last part of a file.

# Show last 'num' lines in file:
tail -n num file

# Show all file since line 'num':
tail -n +num file

# Show last 'num' bytes in file:
tail -c num file

# Keep reading file until Ctrl + C:
tail -f file

# Keep reading file until Ctrl + C, even if the file is rotated:
tail -F file


#-----------------------------------------------------------------------------#
#                                     tr                                      #
#-----------------------------------------------------------------------------#
# Translate characters: run replacements based on single characters and character sets.

# Replace all occurrences of a character in a file, and print the result:
tr find_character replace_character < filename

# Replace all occurrences of a character from another command's output:
echo text | tr find_character replace_character

# Map each character of the first set to the corresponding character of the second set:
tr 'abcd' 'jkmn' < filename

# Delete all occurrences of the specified set of characters from the input:
tr -d 'input_characters' < filename

# Compress a series of identical characters to a single character:
tr -s 'input_characters' < filename

# delete all whitespace
tr -d "[:space:]"

# Translate the contents of a file to upper-case:
tr "[:lower:]" "[:upper:]" < filename

# Strip out non-printable characters from a file:
tr -cd "[:print:]" < filename

#replace : with new line
echo $PATH|tr ":" "\n" #equivalent with:
echo $PATH|tr -t ":" \n

#remove all occurance of "ab"
echo aabbcc |tr -d "ab"
#ouput: cc

#complement "aa"
echo aabbccd |tr -c "aa" 1
#output: aa11111 without new line
#tip: Complement meaning keep aa,all others are replaced with 1

#complement "ab\n"
echo aabbccd |tr -c "ab\n" 1
#output: aabb111 with new line

#Preserve all alpha(-c). ":-[:digit:] etc" will be translated to "\n". sequeeze mode.
echo $PATH|tr -cs "[:alpha:]" "\n"

#ordered list to unordered list
echo "1. /usr/bin\n2. /bin" |tr -cs " /[:alpha:]\n" "+"



#=============================================================================#
#                                                                             #
#                    888                        888                           #
#                    888                        888                           #
#                    888                        888                           #
#                    88888b.   8888b.  .d8888b  88888b.                       #
#                    888 "88b     "88b 88K      888 "88b                      #
#                    888  888 .d888888 "Y8888b. 888  888                      #
#                    888 d88P 888  888      X88 888  888                      #
#                    88888P"  "Y888888  88888P' 888  888                      #
#                                                                             #
#         _                      _           _                     _          #
#        | |                    | |         | |                   | |         #
#   ___  | |__     ___    __ _  | |_   ___  | |__     ___    ___  | |_        #
#  / __| | '_ \   / _ \  / _` | | __| / __| | '_ \   / _ \  / _ \ | __|       #
# | (__  | | | | |  __/ | (_| | | |_  \__ \ | | | | |  __/ |  __/ | |_        #
#  \___| |_| |_|  \___|  \__,_|  \__| |___/ |_| |_|  \___|  \___|  \__|       #
#                                                                             #
# Author: J. Le Coupenec
# https://github.com/LeCoupa/awesome-cheatsheets



##############################################################################
# SHORTCUTS
##############################################################################


CTRL+A  # move to beginning of line
CTRL+B  # moves backward one character
CTRL+C  # halts the current command
CTRL+D  # deletes one character backward or logs out of current session, similar to exit
CTRL+E  # moves to end of line
CTRL+F  # moves forward one character
CTRL+G  # aborts the current editing command and ring the terminal bell
CTRL+J  # same as RETURN
CTRL+K  # deletes (kill) forward to end of line
CTRL+L  # clears screen and redisplay the line
CTRL+M  # same as RETURN
CTRL+N  # next line in command history
CTRL+O  # same as RETURN, then displays next line in history file
CTRL+P  # previous line in command history
CTRL+R  # searches backward
CTRL+S  # searches forward
CTRL+T  # transposes two characters
CTRL+U  # kills backward from point to the beginning of line
CTRL+V  # makes the next character typed verbatim
CTRL+W  # kills the word behind the cursor
CTRL+X  # lists the possible filename completions of the current word
CTRL+Y  # retrieves (yank) last item killed
CTRL+Z  # stops the current command, resume with fg in the foreground or bg in the background

ALT+B   # moves backward one word
ALT+D   # deletes next word
ALT+F   # moves forward one word

DELETE  # deletes one character backward
!!      # repeats the last command
exit    # logs out of current session


##############################################################################
# BASH BASICS
##############################################################################

env                 # displays all environment variables

echo $SHELL         # displays the shell you're using
echo $BASH_VERSION  # displays bash version

bash                # if you want to use bash (type exit to go back to your previously opened shell)
whereis bash        # finds out where bash is on your system
which bash          # finds out which program is executed as 'bash' (default: /bin/bash, can change across environments)

clear               # clears content on window (hide displayed lines)


##############################################################################
# FILE COMMANDS
##############################################################################


ls                            # lists your files in current directory, ls <dir> to print files in a specific directory
ls -l                         # lists your files in 'long format', which contains the exact size of the file, who owns the file and who has the right to look at it, and when it was last modified
ls -a                         # lists all files, including hidden files (name beginning with '.')
ln -s <filename> <link>       # creates symbolic link to file
touch <filename>              # creates or updates (edit) your file
cat <filename>                # prints file raw content (will not be interpreted)
any_command > <filename>      # '>' is used to perform redirections, it will set any_command's stdout to file instead of "real stdout" (generally /dev/stdout)
more <filename>               # shows the first part of a file (move with space and type q to quit)
head <filename>               # outputs the first lines of file (default: 10 lines)
tail <filename>               # outputs the last lines of file (useful with -f option) (default: 10 lines)
vim <filename>                # opens a file in VIM (VI iMproved) text editor, will create it if it doesn't exist
mv <filename1> <dest>         # moves a file to destination, behavior will change based on 'dest' type (dir: file is placed into dir; file: file will replace dest (tip: useful for renaming))
cp <filename1> <dest>         # copies a file
rm <filename>                 # removes a file
diff <filename1> <filename2>  # compares files, and shows where they differ
wc <filename>                 # tells you how many lines, words and characters there are in a file. Use -lwc (lines, word, character) to ouput only 1 of those informations
chmod -options <filename>     # lets you change the read, write, and execute permissions on your files (more infos: SUID, GUID)
gzip <filename>               # compresses files using gzip algorithm
gunzip <filename>             # uncompresses files compressed by gzip
gzcat <filename>              # lets you look at gzipped file without actually having to gunzip it
lpr <filename>                # prints the file
lpq                           # checks out the printer queue
lprm <jobnumber>              # removes something from the printer queue
genscript                     # converts plain text files into postscript for printing and gives you some options for formatting
dvips <filename>              # prints .dvi files (i.e. files produced by LaTeX)
grep <pattern> <filenames>    # looks for the string in the files
grep -r <pattern> <dir>       # search recursively for pattern in directory


##############################################################################
# DIRECTORY COMMANDS
##############################################################################


mkdir <dirname>  # makes a new directory
cd               # changes to home
cd <dirname>     # changes directory
pwd              # tells you where you currently are


##############################################################################
# SSH, SYSTEM INFO & NETWORK COMMANDS
##############################################################################


ssh user@host            # connects to host as user
ssh -p <port> user@host  # connects to host on specified port as user
ssh-copy-id user@host    # adds your ssh key to host for user to enable a keyed or passwordless login

whoami                   # returns your username
passwd                   # lets you change your password
quota -v                 # shows what your disk quota is
date                     # shows the current date and time
cal                      # shows the month's calendar
uptime                   # shows current uptime
w                        # displays whois online
finger <user>            # displays information about user
uname -a                 # shows kernel information
man <command>            # shows the manual for specified command
df                       # shows disk usage
du <filename>            # shows the disk usage of the files and directories in filename (du -s give only a total)
last <yourUsername>      # lists your last logins
ps -u yourusername       # lists your processes
kill <PID>               # kills the processes with the ID you gave
killall <processname>    # kill all processes with the name
top                      # displays your currently active processes
bg                       # lists stopped or background jobs ; resume a stopped job in the background
fg                       # brings the most recent job in the foreground
fg <job>                 # brings job to the foreground

ping <host>              # pings host and outputs results
whois <domain>           # gets whois information for domain
dig <domain>             # gets DNS information for domain
dig -x <host>            # reverses lookup host
wget <file>              # downloads file


##############################################################################
# VARIABLES
##############################################################################


varname=value                # defines a variable
varname=value command        # defines a variable to be in the environment of a particular subprocess
echo $varname                # checks a variable's value
echo $$                      # prints process ID of the current shell
echo $!                      # prints process ID of the most recently invoked background job
echo $?                      # displays the exit status of the last command
export VARNAME=value         # defines an environment variable (will be available in subprocesses)

array[0]=valA                # how to define an array
array[1]=valB
array[2]=valC
array=([2]=valC [0]=valA [1]=valB)  # another way
array=(valA valB valC)              # and another

${array[i]}                  # displays array's value for this index. If no index is supplied, array element 0 is assumed
${#array[i]}                 # to find out the length of any element in the array
${#array[@]}                 # to find out how many values there are in the array

declare -a                   # the variables are treaded as arrays
declare -f                   # uses function names only
declare -F                   # displays function names without definitions
declare -i                   # the variables are treaded as integers
declare -r                   # makes the variables read-only
declare -x                   # marks the variables for export via the environment

${varname:-word}             # if varname exists and isn't null, return its value; otherwise return word
${varname:=word}             # if varname exists and isn't null, return its value; otherwise set it word and then return its value
${varname:?message}          # if varname exists and isn't null, return its value; otherwise print varname, followed by message and abort the current command or script
${varname:+word}             # if varname exists and isn't null, return word; otherwise return null
${varname:offset:length}     # performs substring expansion. It returns the substring of $varname starting at offset and up to length characters

${variable#pattern}          # if the pattern matches the beginning of the variable's value, delete the shortest part that matches and return the rest
${variable##pattern}         # if the pattern matches the beginning of the variable's value, delete the longest part that matches and return the rest
${variable%pattern}          # if the pattern matches the end of the variable's value, delete the shortest part that matches and return the rest
${variable%%pattern}         # if the pattern matches the end of the variable's value, delete the longest part that matches and return the rest
${variable/pattern/string}   # the longest match to pattern in variable is replaced by string. Only the first match is replaced
${variable//pattern/string}  # the longest match to pattern in variable is replaced by string. All matches are replaced

${#varname}                  # returns the length of the value of the variable as a character string

*(patternlist)               # matches zero or more occurrences of the given patterns
+(patternlist)               # matches one or more occurrences of the given patterns
?(patternlist)               # matches zero or one occurrence of the given patterns
@(patternlist)               # matches exactly one of the given patterns
!(patternlist)               # matches anything except one of the given patterns

$(UNIX command)              # command substitution: runs the command and returns standard output


##############################################################################
# FUNCTIONS
##############################################################################


# The function refers to passed arguments by position (as if they were positional parameters), that is, $1, $2, and so forth.
# $@ is equal to "$1" "$2"... "$N", where N is the number of positional parameters. $# holds the number of positional parameters.


function functname() {
  shell commands
}

unset -f functname  # deletes a function definition
declare -f          # displays all defined functions in your login session


##############################################################################
# FLOW CONTROLS
##############################################################################


statement1 && statement2  # and operator
statement1 || statement2  # or operator

-a                        # and operator inside a test conditional expression
-o                        # or operator inside a test conditional expression

# STRINGS

str1 = str2               # str1 matches str2
str1 != str2              # str1 does not match str2
str1 < str2               # str1 is less than str2 (alphabetically)
str1 > str2               # str1 is greater than str2 (alphabetically)
-n str1                   # str1 is not null (has length greater than 0)
-z str1                   # str1 is null (has length 0)

# FILES

-a file                   # file exists
-d file                   # file exists and is a directory
-e file                   # file exists; same -a
-f file                   # file exists and is a regular file (i.e., not a directory or other special type of file)
-r file                   # you have read permission
-s file                   # file exists and is not empty
-w file                   # your have write permission
-x file                   # you have execute permission on file, or directory search permission if it is a directory
-N file                   # file was modified since it was last read
-O file                   # you own file
-G file                   # file's group ID matches yours (or one of yours, if you are in multiple groups)
file1 -nt file2           # file1 is newer than file2
file1 -ot file2           # file1 is older than file2

# NUMBERS

-lt                       # less than
-le                       # less than or equal
-eq                       # equal
-ge                       # greater than or equal
-gt                       # greater than
-ne                       # not equal

if condition
then
  statements
[elif condition
  then statements...]
[else
  statements]
fi

for x in {1..10}
do
  statements
done

for name [in list]
do
  statements that can use $name
done

for (( initialisation ; ending condition ; update ))
do
  statements...
done

case expression in
  pattern1 )
    statements ;;
  pattern2 )
    statements ;;
esac

select name [in list]
do
  statements that can use $name
done

while condition; do
  statements
done

until condition; do
  statements
done

##############################################################################
# COMMAND-LINE PROCESSING CYCLE
##############################################################################


# The default order for command lookup is functions, followed by built-ins, with scripts and executables last.
# There are three built-ins that you can use to override this order: `command`, `builtin` and `enable`.

command  # removes alias and function lookup. Only built-ins and commands found in the search path are executed
builtin  # looks up only built-in commands, ignoring functions and commands found in PATH
enable   # enables and disables shell built-ins

eval     # takes arguments and run them through the command-line processing steps all over again


##############################################################################
# INPUT/OUTPUT REDIRECTORS
##############################################################################


cmd1|cmd2  # pipe; takes standard output of cmd1 as standard input to cmd2
< file     # takes standard input from file
> file     # directs standard output to file
>> file    # directs standard output to file; append to file if it already exists
>|file     # forces standard output to file even if noclobber is set
n>|file    # forces output to file from file descriptor n even if noclobber is set
<> file    # uses file as both standard input and standard output
n<>file    # uses file as both input and output for file descriptor n
n>file     # directs file descriptor n to file
n<file     # takes file descriptor n from file
n>>file    # directs file description n to file; append to file if it already exists
n>&        # duplicates standard output to file descriptor n
n<&        # duplicates standard input from file descriptor n
n>&m       # file descriptor n is made to be a copy of the output file descriptor
n<&m       # file descriptor n is made to be a copy of the input file descriptor
&>file     # directs standard output and standard error to file
<&-        # closes the standard input
>&-        # closes the standard output
n>&-       # closes the ouput from file descriptor n
n<&-       # closes the input from file descripor n


##############################################################################
# PROCESS HANDLING
##############################################################################


# To suspend a job, type CTRL+Z while it is running. You can also suspend a job with CTRL+Y.
# This is slightly different from CTRL+Z in that the process is only stopped when it attempts to read input from terminal.
# Of course, to interrupt a job, type CTRL+C.

myCommand &  # runs job in the background and prompts back the shell

jobs         # lists all jobs (use with -l to see associated PID)

fg           # brings a background job into the foreground
fg %+        # brings most recently invoked background job
fg %-        # brings second most recently invoked background job
fg %N        # brings job number N
fg %string   # brings job whose command begins with string
fg %?string  # brings job whose command contains string

kill -l      # returns a list of all signals on the system, by name and number
kill PID     # terminates process with specified PID

ps           # prints a line of information about the current running login shell and any processes running under it
ps -a        # selects all processes with a tty except session leaders

trap cmd sig1 sig2  # executes a command when a signal is received by the script
trap "" sig1 sig2   # ignores that signals
trap - sig1 sig2    # resets the action taken when the signal is received to the default

disown <PID|JID>    # removes the process from the list of jobs

wait                # waits until all background jobs have finished


##############################################################################
# TIPS & TRICKS
##############################################################################


# set an alias
cd; nano .bash_profile
> alias gentlenode='ssh admin@gentlenode.com -p 3404'  # add your alias in .bash_profile

# to quickly go to a specific directory
cd; nano .bashrc
> shopt -s cdable_vars
> export websites="/Users/poo/Documents/websites"

source .bashrc
cd $websites


##############################################################################
# DEBUGGING SHELL PROGRAMS
##############################################################################


bash -n scriptname  # don't run commands; check for syntax errors only
set -o noexec       # alternative (set option in script)

bash -v scriptname  # echo commands before running them
set -o verbose      # alternative (set option in script)

bash -x scriptname  # echo commands after command-line processing
set -o xtrace       # alternative (set option in script)

trap 'echo $varname' EXIT  # useful when you want to print out the values of variables at the point that your script exits

function errtrap {
  es=$?
  echo "ERROR line $1: Command exited with status $es."
}

trap 'errtrap $LINENO' ERR  # is run whenever a command in the surrounding script or function exits with non-zero status

function dbgtrap {
  echo "badvar is $badvar"
}

trap dbgtrap DEBUG  # causes the trap code to be executed before every statement in a function or script
# ...section of code in which the problem occurs...
trap - DEBUG  # turn off the DEBUG trap

function returntrap {
  echo "A return occurred"
}

trap returntrap RETURN  # is executed each time a shell function or a script executed with the . or source commands finishes executing
