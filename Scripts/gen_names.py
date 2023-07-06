#!/usr/bin/env python
# Copyright 2022 Evan Cushing. All rights reserved.
import random

#=============================================================================#
#                                  Word Bank                                  #
#=============================================================================#
# 𝗦𝗢𝗨𝗥𝗖𝗘: https://github.com/goombaio/namegenerator/blob/master/data.go
#-----------------------------------------------------------------------------#


ADJECTIVES = [
    'autumn',
    'hidden',
    'bitter',
    'misty',
    'silent',
    'empty',
    'dry',
    'dark',
    'summer',
    'icy',
    'delicate',
    'quiet',
    'white',
    'cool',
    'spring',
    'winter',
    'patient',
    'twilight',
    'dawn',
    'crimson',
    'wispy',
    'weathered',
    'blue',
    'billowing',
    'broken',
    'cold',
    'damp',
    'falling',
    'frosty',
    'green',
    'long',
    'late',
    'lingering',
    'bold',
    'little',
    'morning',
    'muddy',
    'old',
    'red',
    'rough',
    'still',
    'small',
    'sparkling',
    'throbbing',
    'shy',
    'wandering',
    'withered',
    'wild',
    'black',
    'young',
    'holy',
    'solitary',
    'fragrant',
    'aged',
    'snowy',
    'proud',
    'floral',
    'restless',
    'divine',
    'polished',
    'ancient',
    'purple',
    'lively',
    'nameless',
    'enigmatic',
    'luminescent',
    'delicate',
    'mystical',
    'harmonious',
    'ephemeral',
    'veiled',
    'ethereal',
    'enchanted',
    'shimmering',
    'lush',
    'nimble',
    'fluttering',
    'radiant',
    'glowing',
    'verdant',
    'pristine',
    'dashing',
]


NOUNS = [
    'waterfall',
    'river',
    'breeze',
    'moon',
    'rain',
    'wind',
    'sea',
    'morning',
    'snow',
    'lake',
    'sunset',
    'pine',
    'shadow',
    'leaf',
    'dawn',
    'glitter',
    'forest',
    'hill',
    'cloud',
    'meadow',
    'sun',
    'glade',
    'bird',
    'brook',
    'butterfly',
    'bush',
    'dew',
    'dust',
    'field',
    'fire',
    'flower',
    'firefly',
    'feather',
    'grass',
    'haze',
    'mountain',
    'night',
    'pond',
    'darkness',
    'snowflake',
    'silence',
    'sound',
    'sky',
    'shape',
    'surf',
    'thunder',
    'violet',
    'water',
    'wildflower',
    'wave',
    'water',
    'resonance',
    'sun',
    'wood',
    'dream',
    'cherry',
    'tree',
    'fog',
    'frost',
    'voice',
    'paper',
    'frog',
    'smoke',
    'star',
    'zephyr',
    'aurora',
    'whisper',
    'melody',
    'blossom',
    'secret',
    'reverie',
    'cascade',
    'mirage',
    'blade',
    'tide',
    'stone',
    'glow',
]


#=============================================================================#
#                                  Functions                                  #
#=============================================================================#


def gen_name():
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    name = f"{adjective}-{noun}"
    return name


def gen_names_and_print_them(n=100):
    for _ in range(n):
        name = gen_name()
        print(name)


def main():
    import sys
    n = 100
    if sys.argv[1:]:
        n = int(sys.argv[1])
        assert 1 <= n <= 1000
    gen_names_and_print_them(n)


if __name__ == '__main__':
    main()
