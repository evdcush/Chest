
# -*- coding: utf-8 -*-
"""
    pygments.styles.paraiso_dark
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Paraíso (Dark) by Jan T. Sott

    Pygments template by Jan T. Sott (https://github.com/idleberg)
    Created with Base16 Builder by Chris Kempson
    (https://github.com/chriskempson/base16-builder).

    :copyright: Copyright 2006-2017 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, Text, \
    Number, Operator, Generic, Whitespace, Punctuation, Other, Literal


BACKGROUND = "#16152B"
CURRENT_LINE = "#41323f"
SELECTION = "#4f424c"
FOREGROUND = "#e7e9db"
COMMENT = "#776e71"
RED = "#ef6155"
ORANGE = "#f99b15"
YELLOW = "#fec418"
GREEN = "#48b685"
AQUA = "#5bc4bf"
BLUE = "#06b6ef"
PURPLE = "#815ba4"

#=== pio

GREY = "#9E9D75"
ALABASTER = "#F8F8F2" # WHITE-GREY-WHITE
ROSE = "#FF757E"      # a little more sat than reg rose
OCHRE = "#F0A865"     # DARK-ORANGE-YELLOW
GOLDEN_YELLOW = "#FFEA00"
SEAFOAM    = "#63EBA5" # LIGHT-GREEN-TURQUOISE
DEEP_SKY   = "#2DB7FC" # BLUE-SKYBLUE
PERIWINKLE = "#C2CAFF" # BLUE-LAVENDER-BLUE
WISTERIA   = "#DB9BEB" # WISTERIA-VIOLET-WISTERIA
HELIOTROPE = "#C57AFF" # SAT-WISTERIA

MIDNIGHT_EXPRESS = "#16152B" # DARK-SLATE
TOBACCO = "#665941" # comments

class PiodineStyle(Style):
    default_style = ''

    background_color = MIDNIGHT_EXPRESS
    highlight_color = SELECTION

    styles = {

        # No corresponding class for the following:
        Text:                   ALABASTER,  # class:  ''
        Whitespace:             "",          # class: 'w'
        Error:                  RED,         # class: 'err'
        Other:                  ALABASTER,          # class 'x'

        Comment:                TOBACCO, #GREY,
        Comment.Preproc:        TOBACCO, #GREY,
        Comment.Single:         TOBACCO, #GREY,
        Comment.Special:        TOBACCO, #GREY,
        Comment.Multiline:      TOBACCO, #GREY,
        Number:                 ROSE,
        Number.Bin:             ROSE,
        Number.Float:           ROSE,
        Number.Hex:             ROSE,
        Number.Integer:         ROSE,
        Number.Integer.Long:    ROSE,
        Number.Oct:             ROSE,
        String.Escape:          SEAFOAM,
        Keyword.Constant:       SEAFOAM,
        Keyword:                OCHRE + ' bold',
        Keyword.Declaration:    OCHRE + ' italic',
        Keyword.Namespace:      WISTERIA,
        Keyword.Pseudo:         WISTERIA,
        Keyword.Reserved:       WISTERIA,
        Keyword.Type:           OCHRE,
        Literal:                ALABASTER,
        Literal.Date:           ALABASTER,
        String:                 GOLDEN_YELLOW,
        String.Backtick:        GOLDEN_YELLOW,
        String.Char:            GOLDEN_YELLOW,
        String.Doc:             GOLDEN_YELLOW,
        String.Double:          GOLDEN_YELLOW,
        String.Heredoc:         GOLDEN_YELLOW,
        String.Interpol:        GOLDEN_YELLOW,
        String.Other:           GOLDEN_YELLOW,
        String.Regex:           GOLDEN_YELLOW,
        String.Single:          GOLDEN_YELLOW,
        String.Symbol:          GOLDEN_YELLOW,
        Name.Attribute:         SEAFOAM,
        Name.Builtin:           OCHRE + ' italic',
        Name.Builtin.Pseudo:    SEAFOAM,
        Name.Class:             PERIWINKLE + ' underline',
        Name.Constant:          SEAFOAM,
        Name.Decorator:         PERIWINKLE,
        Name.Entity:            ALABASTER,
        Name.Exception:         PERIWINKLE,
        Name.Function:          DEEP_SKY,
        Name.Tag:               HELIOTROPE + ' bold',
        Name.Variable:          SEAFOAM, # + ' italic'
        Name.Variable.Class:    SEAFOAM + ' italic',
        Name.Variable.Global:   SEAFOAM + ' italic',
        Name.Variable.Instance: SEAFOAM + ' italic',
        Operator.Word:          WISTERIA + ' bold',
        Operator:               WISTERIA,
        Name:                   ALABASTER,
        Name.Label:             ALABASTER,
        Name.Namespace:         ALABASTER + ' bold',
        Name.Other:             ALABASTER,
        Punctuation:            ALABASTER,
        Generic:                   ALABASTER,                    # class: 'g'
        Generic.Deleted:           RED,                   # class: 'gd',
        Generic.Emph:              "italic",              # class: 'ge'
        Generic.Error:             "",                    # class: 'gr'
        Generic.Heading:           FOREGROUND + ' italic',  # class: 'gh'
        Generic.Inserted:          GREEN,                 # class: 'gi'
        Generic.Output:            "",                    # class: 'go'
        Generic.Prompt:            COMMENT + ' bold',     # class: 'gp'
        Generic.Strong:            "bold",                # class: 'gs'
        Generic.Subheading:        AQUA + ' bold',        # class: 'gu'
        Generic.Traceback:         ALABASTER,

        }


