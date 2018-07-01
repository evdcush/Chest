# If you come from bash you might have to change your $PATH.
# export PATH=$HOME/bin:/usr/local/bin:$PATH

# Path to your oh-my-zsh installation.
  export ZSH="/home/evan/.oh-my-zsh"

# Set name of the theme to load. Optionally, if you set this to "random"
# it'll load a random theme each time that oh-my-zsh is loaded.
# See https://github.com/robbyrussell/oh-my-zsh/wiki/Themes
ZSH_THEME="robbyrussell"

# Set list of themes to load
# Setting this variable when ZSH_THEME=random
# cause zsh load theme from this variable instead of
# looking in ~/.oh-my-zsh/themes/
# An empty array have no effect
# ZSH_THEME_RANDOM_CANDIDATES=( "robbyrussell" "agnoster" )

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion. Case
# sensitive completion must be off. _ and - will be interchangeable.
# HYPHEN_INSENSITIVE="true"

# Uncomment the following line to disable bi-weekly auto-update checks.
# DISABLE_AUTO_UPDATE="true"

# Uncomment the following line to change how often to auto-update (in days).
# export UPDATE_ZSH_DAYS=13

# Uncomment the following line to disable colors in ls.
# DISABLE_LS_COLORS="true"

# Uncomment the following line to disable auto-setting terminal title.
# DISABLE_AUTO_TITLE="true"

# Uncomment the following line to enable command auto-correction.
# ENABLE_CORRECTION="true"

# Uncomment the following line to display red dots whilst waiting for completion.
# COMPLETION_WAITING_DOTS="true"

# Uncomment the following line if you want to disable marking untracked files
# under VCS as dirty. This makes repository status check for large repositories
# much, much faster.
# DISABLE_UNTRACKED_FILES_DIRTY="true"

# Uncomment the following line if you want to change the command execution time
# stamp shown in the history command output.
# The optional three formats: "mm/dd/yyyy"|"dd.mm.yyyy"|"yyyy-mm-dd"
# HIST_STAMPS="mm/dd/yyyy"

# Would you like to use another custom folder than $ZSH/custom?
# ZSH_CUSTOM=/path/to/new-custom-folder

# Which plugins would you like to load? (plugins can be found in ~/.oh-my-zsh/plugins/*)
# Custom plugins may be added to ~/.oh-my-zsh/custom/plugins/
# Example format: plugins=(rails git textmate ruby lighthouse)
# Add wisely, as too many plugins slow down shell startup.
plugins=(
  git
  sublime
  sudo
)

source $ZSH/oh-my-zsh.sh

# User configuration

# export MANPATH="/usr/local/man:$MANPATH"

# You may need to manually set your language environment
# export LANG=en_US.UTF-8

# Preferred editor for local and remote sessions
# if [[ -n $SSH_CONNECTION ]]; then
#   export EDITOR='vim'
# else
#   export EDITOR='mvim'
# fi

# Compilation flags
# export ARCHFLAGS="-arch x86_64"

# ssh
# export SSH_KEY_PATH="~/.ssh/rsa_id"

# Set personal aliases, overriding those provided by oh-my-zsh libs,
# plugins, and themes. Aliases can be placed here, though oh-my-zsh
# users are encouraged to define aliases within the ZSH_CUSTOM folder.
# For a full list of active aliases, run `alias`.
#
# Example aliases
# alias zshconfig="mate ~/.zshrc"
# alias ohmyzsh="mate ~/.oh-my-zsh"
#

##### USER SHIT #####

# Pyenv
export PATH="/home/evan/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Z jump
. /home/evan/Soft/Installed/Utils/z_jump/z.sh

# MKL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib



### FUNCTIONS
# Extra many types of compressed packages
# Credit: http://nparikh.org/notes/zshrc.txt
function extract() {
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

# List dir, sorted by file type
function lk(){
    ls -lX;
}

# list top dir file sizes, sorted
function duh(){
    du -h --max-depth=1 | sort -h;
}

# Search man
fman () {
    # Mnemonic: Full-text MAN
    # Search the entire text of all man pages for the 
    # specified term, then print out the matching pages
    # locations.
    man -wK "${@}"
}

# Rename files to lowercase+underscore, removing spaces and quotes
mvlu () {
    # Mnemonic: Move Verbose Lowercase Underscore 
    # Rename files to lowercase, replacing spaces with 
    # underscores, and deleting quotes and control 
    # characters.
    for filename in "${@}"; do
        mv --verbose -- \
            "${filename}" \
            "$(echo "${filename}" \
                | tr '[:upper:]' '[:lower:]' \
                | tr '[:blank:]' '_' \
                | tr -d \'\" \
                | tr -d '[:cntrl:]'
              )"
    done
}

# Show local IP
lip () {
    # Mnemonic: Local IP.
    ip --color -s -s address show "${@}"
}

# show external IP
xip () {
    # Mnemonic: eXternal IP.
    echo "External IP Addresses"
    dig +short myip.opendns.com @resolver1.opendns.com
}

