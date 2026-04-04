#!/usr/bin/env python3
"""Unicode text enfancier: CLI + tiny Tkinter GUI.

Pure standard-library implementation, with best-effort grapheme handling and a
curated set of Unicode substitution styles.

NB:
- made by bot
- python "port" of `unicode_gen.js` (simple unicode text style gen)
"""

import argparse
import sys
import unicodedata
from typing import Callable, Dict, List, Optional


ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"
ASCII_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
ZWSP = "\u200b"

DEFAULT_SAMPLE = "Text Enfancier!"


def _chars(*codepoints: int) -> str:
    return "".join(chr(cp) for cp in codepoints)


def _range_chars(start: int, count: int) -> str:
    return "".join(chr(start + i) for i in range(count))


def _map_seq(
    lower: Optional[str] = None,
    upper: Optional[str] = None,
    digits: Optional[str] = None,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if lower:
        mapping.update(zip(ASCII_LOWER, lower))
    if upper:
        mapping.update(zip(ASCII_UPPER, upper))
    if digits:
        mapping.update(zip(DIGITS, digits))
    if extra:
        mapping.update(extra)
    return mapping


def _is_combining(ch: str) -> bool:
    return unicodedata.category(ch).startswith("M") or ch in {"\u200d", "\u200c"}


def _is_regional_indicator(ch: str) -> bool:
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def split_graphemes(text: str) -> List[str]:
    """Best-effort grapheme splitter without third-party deps."""
    clusters: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        cluster = ch
        i += 1

        if (
            _is_regional_indicator(ch)
            and i < len(text)
            and _is_regional_indicator(text[i])
        ):
            cluster += text[i]
            i += 1

        while i < len(text):
            nxt = text[i]
            if (
                _is_combining(nxt)
                or unicodedata.category(nxt) == "Cf"
                or 0xFE00 <= ord(nxt) <= 0xFE0F
                or 0x1F3FB <= ord(nxt) <= 0x1F3FF
            ):
                cluster += nxt
                i += 1
                continue
            break

        clusters.append(cluster)
    return clusters


def _map_char(ch: str, mapping: Dict[str, str]) -> str:
    if ch in mapping:
        return mapping[ch]
    upper = ch.upper()
    if upper in mapping:
        return mapping[upper]
    lower = ch.lower()
    if lower in mapping:
        return mapping[lower]
    return ch


def _transform_clusters(
    text: str,
    *,
    letter_map: Optional[Dict[str, str]] = None,
    digit_map: Optional[Dict[str, str]] = None,
    multi_digit_map: Optional[Dict[str, str]] = None,
    extra_map: Optional[Dict[str, str]] = None,
    decorator: Optional[str] = None,
    reverse_output: bool = False,
    custom_digit_run: Optional[Callable[[str], Optional[str]]] = None,
    regional_indicator_spacing: bool = False,
) -> str:
    clusters = split_graphemes(text)
    if reverse_output:
        clusters = list(reversed(clusters))

    out: List[str] = []
    i = 0
    while i < len(clusters):
        cluster = clusters[i]

        if custom_digit_run and len(cluster) == 1 and cluster.isdigit():
            j = i
            while j < len(clusters) and len(clusters[j]) == 1 and clusters[j].isdigit():
                j += 1
            repl = custom_digit_run("".join(clusters[i:j]))
            if repl is not None:
                out.append(repl)
                i = j
                continue

        if multi_digit_map and len(cluster) == 1 and cluster.isdigit():
            j = i
            while j < len(clusters) and len(clusters[j]) == 1 and clusters[j].isdigit():
                j += 1
            run = "".join(clusters[i:j])
            if run in multi_digit_map:
                out.append(multi_digit_map[run])
                i = j
                continue
            if digit_map:
                out.extend(_map_char(c, digit_map) for c in run)
                i = j
                continue

        if decorator and len(cluster) == 1 and not cluster.isspace():
            out.append(cluster + decorator)
            i += 1
            continue

        if len(cluster) == 1:
            ch = cluster
            if extra_map and ch in extra_map:
                out.append(extra_map[ch])
            elif letter_map and (
                ch.isalpha()
                or ch in letter_map
                or ch.upper() in letter_map
                or ch.lower() in letter_map
            ):
                out.append(_map_char(ch, letter_map))
            elif digit_map and ch.isdigit():
                out.append(_map_char(ch, digit_map))
            else:
                out.append(ch)
        else:
            out.append(cluster)

        i += 1

    text_out = "".join(out)
    if regional_indicator_spacing:
        spaced: List[str] = []
        prev_ri = False
        for ch in text_out:
            cur_ri = _is_regional_indicator(ch)
            if prev_ri and cur_ri:
                spaced.append(ZWSP)
            spaced.append(ch)
            prev_ri = cur_ri
        text_out = "".join(spaced)
    return text_out


def _reverse_map(mapping: Dict[str, str]) -> Dict[str, str]:
    return {v: k for k, v in mapping.items()}


def _roman_from_int(n: int, lowercase: bool = False) -> str:
    if n <= 0 or n > 3999:
        return str(n)
    numerals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = []
    for value, glyph in numerals:
        while n >= value:
            out.append(glyph)
            n -= value
    roman = "".join(out)
    return roman.lower() if lowercase else roman


def _convert_digit_run_to_roman(run: str, lowercase: bool = False) -> Optional[str]:
    try:
        n = int(run)
    except ValueError:
        return None
    return _roman_from_int(n, lowercase=lowercase)


def _build_styles() -> Dict[str, Callable[[str], str]]:
    styles: Dict[str, Callable[[str], str]] = {}

    circled_number_chars = (
        [0x24EA]
        + list(range(0x2460, 0x2474))
        + list(range(0x3251, 0x3260))
        + list(range(0x32B1, 0x32C0))
    )
    circled_number_map = {str(i): chr(cp) for i, cp in enumerate(circled_number_chars)}
    black_circled_number_chars = (
        [0x24FF] + list(range(0x2776, 0x2780)) + list(range(0x24EB, 0x24F5))
    )
    black_circled_number_map = {
        str(i): chr(cp) for i, cp in enumerate(black_circled_number_chars)
    }

    circled_letters = _map_seq(
        lower=_chars(*range(0x24D0, 0x24EA)),
        upper=_chars(*range(0x24B6, 0x24D0)),
        digits=_chars(*range(0x24EA, 0x24EA + 1))
        + _chars(*range(0x2460, 0x2474))
        + _chars(*range(0x3251, 0x3260))
        + _chars(*range(0x32B1, 0x32C0)),
    )
    black_circled_letters = _map_seq(
        upper=_chars(*range(0x1F150, 0x1F16A)),
        digits=_chars(0x24FF, *range(0x2776, 0x2780), *range(0x24EB, 0x24F5)),
    )
    fullwidth_letters = _map_seq(
        lower=_chars(*range(0xFF41, 0xFF5B)),
        upper=_chars(*range(0xFF21, 0xFF3B)),
        digits=_chars(*range(0xFF10, 0xFF1A)),
        extra={
            " ": "\u3000",
            "!": "！",
            "?": "？",
            ".": "．",
            ",": "，",
            ":": "：",
            ";": "；",
            "-": "－",
            "_": "＿",
            "(": "（",
            ")": "）",
        },
    )
    bold = _map_seq(
        lower=_range_chars(0x1D41A, 26),
        upper=_range_chars(0x1D400, 26),
        digits=_range_chars(0x1D7CE, 10),
    )
    italic = _map_seq(lower=_range_chars(0x1D44E, 26), upper=_range_chars(0x1D434, 26))
    bold_italic = _map_seq(
        lower=_range_chars(0x1D482, 26),
        upper=_range_chars(0x1D468, 26),
        digits=_range_chars(0x1D7CE, 10),
    )
    script = _map_seq(
        lower=_chars(
            0x1D4B6,
            0x1D4B7,
            0x1D4B8,
            0x1D4B9,
            0x1D452,
            0x1D4BB,
            0x1D454,
            0x1D4BD,
            0x1D4BE,
            0x1D4BF,
            0x1D4C0,
            0x1D4C1,
            0x1D4C2,
            0x1D4C3,
            0x1D45C,
            0x1D4C5,
            0x1D4C6,
            0x1D4C7,
            0x1D4C8,
            0x1D4C9,
            0x1D4CA,
            0x1D4CB,
            0x1D4CC,
            0x1D4CD,
            0x1D4CE,
            0x1D4CF,
        ),
        upper=_chars(
            0x1D49C,
            0x1D435,
            0x1D49E,
            0x1D49F,
            0x1D438,
            0x1D439,
            0x1D4A2,
            0x1D43B,
            0x1D43C,
            0x1D4A5,
            0x1D4A6,
            0x1D43F,
            0x1D440,
            0x1D4A9,
            0x1D4AA,
            0x1D4AB,
            0x1D4AC,
            0x1D445,
            0x1D4AE,
            0x1D4AF,
            0x1D4B0,
            0x1D4B1,
            0x1D4B2,
            0x1D4B3,
            0x1D4B4,
            0x1D4B5,
        ),
    )
    bold_script = _map_seq(
        lower=_range_chars(0x1D4EA, 26),
        upper=_range_chars(0x1D4D0, 26),
        digits=_range_chars(0x1D7CE, 10),
    )
    fraktur = _map_seq(
        lower=_range_chars(0x1D51E, 26),
        upper=_chars(
            0x1D504,
            0x1D505,
            0x212D,
            0x1D507,
            0x1D508,
            0x1D509,
            0x1D50A,
            0x210C,
            0x2111,
            0x1D50D,
            0x1D50E,
            0x1D50F,
            0x1D510,
            0x1D511,
            0x1D512,
            0x1D513,
            0x1D514,
            0x211C,
            0x1D516,
            0x1D517,
            0x1D518,
            0x1D519,
            0x1D51A,
            0x1D51B,
            0x1D51C,
            0x2128,
        ),
    )
    bold_fraktur = _map_seq(
        lower=_range_chars(0x1D586, 26),
        upper=_range_chars(0x1D56C, 26),
        digits=_range_chars(0x1D7CE, 10),
    )
    double_struck = _map_seq(
        lower=_range_chars(0x1D552, 26),
        upper=_chars(
            0x1D538,
            0x1D539,
            0x2102,
            0x1D53B,
            0x1D53C,
            0x1D53D,
            0x1D53E,
            0x210D,
            0x1D540,
            0x1D541,
            0x1D542,
            0x1D543,
            0x1D544,
            0x2115,
            0x1D546,
            0x2119,
            0x211A,
            0x211D,
            0x1D54A,
            0x1D54B,
            0x1D54C,
            0x1D54D,
            0x1D54E,
            0x1D54F,
            0x1D550,
            0x2124,
        ),
        digits=_range_chars(0x1D7D8, 10),
    )
    monospace = _map_seq(
        lower=_range_chars(0x1D68A, 26),
        upper=_range_chars(0x1D670, 26),
        digits=_range_chars(0x1D7F6, 10),
    )
    sans = _map_seq(
        lower=_range_chars(0x1D5BA, 26),
        upper=_range_chars(0x1D5A0, 26),
        digits=_range_chars(0x1D7E2, 10),
    )
    sans_bold = _map_seq(
        lower=_range_chars(0x1D5EE, 26),
        upper=_range_chars(0x1D5D4, 26),
        digits=_range_chars(0x1D7EC, 10),
    )
    sans_italic = _map_seq(
        lower=_range_chars(0x1D622, 26), upper=_range_chars(0x1D608, 26)
    )
    sans_bold_italic = _map_seq(
        lower=_range_chars(0x1D656, 26), upper=_range_chars(0x1D63C, 26)
    )

    styles["plain"] = lambda s: s
    styles["circled"] = lambda s: _transform_clusters(
        s,
        letter_map=circled_letters,
        digit_map={str(i): chr(cp) for i, cp in enumerate(circled_number_chars[:10])},
        multi_digit_map=circled_number_map,
    )
    styles["black_circled"] = lambda s: _transform_clusters(
        s,
        letter_map=black_circled_letters,
        digit_map={
            str(i): chr(cp) for i, cp in enumerate(black_circled_number_chars[:10])
        },
        multi_digit_map=black_circled_number_map,
    )
    styles["fullwidth"] = lambda s: _transform_clusters(
        s,
        letter_map=fullwidth_letters,
        digit_map=fullwidth_letters,
        extra_map=fullwidth_letters,
    )
    styles["bold"] = lambda s: _transform_clusters(s, letter_map=bold, digit_map=bold)
    styles["italic"] = lambda s: _transform_clusters(s, letter_map=italic)
    styles["bold_italic"] = lambda s: _transform_clusters(
        s,
        letter_map=bold_italic,
        digit_map=_map_seq(digits=_range_chars(0x1D7CE, 10)),
    )
    styles["script"] = lambda s: _transform_clusters(s, letter_map=script)
    styles["bold_script"] = lambda s: _transform_clusters(
        s,
        letter_map=bold_script,
        digit_map=_map_seq(digits=_range_chars(0x1D7CE, 10)),
    )
    styles["fraktur"] = lambda s: _transform_clusters(s, letter_map=fraktur)
    styles["bold_fraktur"] = lambda s: _transform_clusters(
        s,
        letter_map=bold_fraktur,
        digit_map=_map_seq(digits=_range_chars(0x1D7CE, 10)),
    )
    styles["double_struck"] = lambda s: _transform_clusters(
        s,
        letter_map=double_struck,
        digit_map=_map_seq(digits=_range_chars(0x1D7D8, 10)),
    )
    styles["monospace"] = lambda s: _transform_clusters(
        s,
        letter_map=monospace,
        digit_map=_map_seq(digits=_range_chars(0x1D7F6, 10)),
    )
    styles["sans"] = lambda s: _transform_clusters(
        s, letter_map=sans, digit_map=_map_seq(digits=_range_chars(0x1D7E2, 10))
    )
    styles["sans_bold"] = lambda s: _transform_clusters(
        s,
        letter_map=sans_bold,
        digit_map=_map_seq(digits=_range_chars(0x1D7EC, 10)),
    )
    styles["sans_italic"] = lambda s: _transform_clusters(s, letter_map=sans_italic)
    styles["sans_bold_italic"] = lambda s: _transform_clusters(
        s, letter_map=sans_bold_italic
    )

    white_sq = _map_seq(upper=_chars(*range(0x1F130, 0x1F14A)))
    black_sq = _map_seq(
        upper=_chars(*range(0x1F170, 0x1F18A)), extra={"!": "❗", "?": "❓"}
    )
    regional = _map_seq(upper=_chars(*range(0x1F1E6, 0x1F200)))
    styles["squared"] = lambda s: _transform_clusters(s, letter_map=white_sq)
    styles["black_squared"] = lambda s: _transform_clusters(
        s, letter_map=black_sq, extra_map=black_sq
    )
    styles["regional_indicator"] = lambda s: _transform_clusters(
        s, letter_map=regional, regional_indicator_spacing=True
    )
    styles["small_caps"] = styles["regional_indicator"]

    parenthesized_numbers = {
        str(i): c for i, c in zip(range(1, 21), _chars(*range(0x2474, 0x2488)))
    }
    styles["parenthesized"] = lambda s: _transform_clusters(
        s,
        multi_digit_map=parenthesized_numbers,
        digit_map={str(i): c for i, c in enumerate(DIGITS)},
    )
    styles["superscript"] = lambda s: _transform_clusters(
        s,
        letter_map=_map_seq(
            lower=_chars(
                0x1D43,
                0x1D47,
                0x1D9C,
                0x1D48,
                0x1D49,
                0x1DA0,
                0x1D4D,
                0x02B0,
                0x2071,
                0x02B2,
                0x1D4F,
                0x02E1,
                0x1D50,
                0x207F,
                0x1D52,
                0x1D56,
                0x06F9,
                0x02B3,
                0x02E2,
                0x1D57,
                0x1D58,
                0x1D5B,
                0x02B7,
                0x02E3,
                0x02B8,
                0x1DBB,
            ),
            upper=_chars(
                0x1D2C,
                0x1D2E,
                0x1466,
                0x1D30,
                0x1D31,
                0x2E01,
                0x1D33,
                0x1D34,
                0x1D35,
                0x1D36,
                0x1D37,
                0x1D38,
                0x1D39,
                0x1D3A,
                0x1D3C,
                0x1D3E,
                0x06F9,
                0x1D3F,
                0x02E2,
                0x1D40,
                0x1D41,
                0x1D42,
                0x02E3,
                0x1D5D,
                0x1D21,
                0x1D5C,
            ),
        ),
        digit_map=_map_seq(
            digits=_chars(
                0x2070,
                0x00B9,
                0x00B2,
                0x00B3,
                0x2074,
                0x2075,
                0x2076,
                0x2077,
                0x2078,
                0x2079,
            )
        ),
    )
    styles["subscript"] = lambda s: _transform_clusters(
        s,
        letter_map=_map_seq(
            lower=_chars(
                0x2090,
                0x1D66,
                0xA700,
                ord("d"),
                0x2091,
                ord("f"),
                0x2089,
                0x2095,
                0x1D62,
                0x2C7C,
                0x2096,
                0x2097,
                0x2098,
                0x2099,
                0x2092,
                0x209A,
                ord("q"),
                0x1D63,
                0x209B,
                0x209C,
                0x1D64,
                0x1D65,
                ord("w"),
                0x2093,
                0x1D67,
                0x2082,
            ),
            extra={"+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎", "~": "˷"},
        ),
        digit_map=_map_seq(
            digits=_chars(
                0x2080,
                0x2081,
                0x2082,
                0x2083,
                0x2084,
                0x2085,
                0x2086,
                0x2087,
                0x2088,
                0x2089,
            )
        ),
    )

    roman_upper = lambda s: _transform_clusters(
        s,
        custom_digit_run=lambda run: _convert_digit_run_to_roman(run, lowercase=False),
    )
    roman_lower = lambda s: _transform_clusters(
        s, custom_digit_run=lambda run: _convert_digit_run_to_roman(run, lowercase=True)
    )
    styles["roman_upper"] = roman_upper
    styles["roman_lower"] = roman_lower

    styles["unique_1"] = lambda s: _transform_clusters(
        s,
        digit_map={
            str(i): c
            for i, c in enumerate(
                _chars(
                    0x03B8,
                    0x02E6,
                    0x03E8,
                    0x0545,
                    0x03E4,
                    0x01BC,
                    0x03B4,
                    0x10487,
                    0x03D0,
                    0x018D,
                )
            )
        },
    )
    styles["unique_2"] = lambda s: _transform_clusters(
        s,
        digit_map={
            str(i): c
            for i, c in enumerate(
                _chars(
                    0x25CB,
                    0x2951,
                    0x057B,
                    0x10F3,
                    0x02AE,
                    0x0495,
                    0x03ED,
                    0x204A,
                    0x10D6,
                    0x1D690,
                )
            )
        },
    )

    mirror_map = {
        "a": "ɐ",
        "b": "q",
        "c": "ɔ",
        "d": "p",
        "e": "ǝ",
        "f": "ɟ",
        "g": "ƃ",
        "h": "ɥ",
        "i": "ᴉ",
        "j": "ɾ",
        "k": "ʞ",
        "l": "ן",
        "m": "ɯ",
        "n": "u",
        "o": "o",
        "p": "d",
        "q": "b",
        "r": "ɹ",
        "s": "s",
        "t": "ʇ",
        "u": "n",
        "v": "ʌ",
        "w": "ʍ",
        "x": "x",
        "y": "ʎ",
        "z": "z",
        "A": "∀",
        "B": "𐐒",
        "C": "Ɔ",
        "D": "◖",
        "E": "Ǝ",
        "F": "Ⅎ",
        "G": "פ",
        "H": "H",
        "I": "I",
        "J": "ſ",
        "K": "⋊",
        "L": "˥",
        "M": "W",
        "N": "N",
        "O": "O",
        "P": "Ԁ",
        "Q": "Ό",
        "R": "ᴚ",
        "S": "S",
        "T": "┴",
        "U": "∩",
        "V": "Λ",
        "W": "M",
        "X": "X",
        "Y": "⅄",
        "Z": "Z",
        "1": "Ɩ",
        "2": "ᄅ",
        "3": "Ɛ",
        "4": "ㄣ",
        "5": "ϛ",
        "6": "9",
        "7": "ㄥ",
        "8": "8",
        "9": "6",
        "0": "0",
        "!": "¡",
        "?": "¿",
        "(": ")",
        ")": "(",
        "[": "]",
        "]": "[",
        "{": "}",
        "}": "{",
        "<": ">",
        ">": "<",
        "/": "\\",
        "\\": "/",
    }
    styles["mirrored"] = lambda s: _transform_clusters(s, extra_map=mirror_map)
    styles["upside_down"] = lambda s: _transform_clusters(
        s, extra_map=mirror_map, reverse_output=True
    )

    accents = {
        "acute": "\u0301",
        "grave": "\u0300",
        "diaeresis": "\u0308",
        "dot": "\u0307",
        "underline": "\u0332",
        "strike": "\u0338",
        "slash": "\u0337",
        "overline": "\u0305",
        "macron": "\u0304",
        "tilde": "\u0303",
        "ring": "\u030a",
        "double_acute": "\u030b",
        "bottom_double_dotted": "\u0324",
        "top_double_dotted": "\u0308",
        "quadruple_dotted": "\u20db",
    }
    for name, mark in accents.items():
        styles[name] = lambda s, mark=mark: _transform_clusters(s, decorator=mark)

    return styles


STYLES = _build_styles()

ALIASES = {
    "all": "plain",
    "white_circled": "circled",
    "white-circled": "circled",
    "black_circled": "black_circled",
    "black-circled": "black_circled",
    "math_fraktur": "fraktur",
    "math-fraktur": "fraktur",
    "blackletter": "fraktur",
    "black_letter": "fraktur",
    "negative_squared": "black_squared",
    "negative-squared": "black_squared",
    "regional-indicator": "regional_indicator",
    "small-caps": "small_caps",
    "upside-down": "upside_down",
    "reverse": "mirrored",
    "reversed": "upside_down",
    "bold-italic": "bold_italic",
    "bold-script": "bold_script",
    "bold-fraktur": "bold_fraktur",
    "double-struck": "double_struck",
    "sans-serif": "sans",
    "sans-serif-bold": "sans_bold",
    "sans-serif-italic": "sans_italic",
    "sans-serif-bold-italic": "sans_bold_italic",
    "white-squared": "squared",
    "black-squared": "black_squared",
}


def normalize_style_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return ALIASES.get(key, key)


def available_styles() -> List[str]:
    return list(STYLES.keys())


def transform(style: str, text: str) -> str:
    style_key = normalize_style_name(style)
    if style_key not in STYLES:
        raise KeyError(style)
    return STYLES[style_key](text)


def transform_all(text: str, labels: bool = False) -> str:
    lines = []
    for name in STYLES:
        out = STYLES[name](text)
        lines.append(f"{name}: {out}" if labels else out)
    return "\n".join(lines)


def _get_input_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data:
            return data.rstrip("\n")
    return DEFAULT_SAMPLE


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="enfancy.py", description="Unicode text enfancier")
    p.add_argument("text", nargs="?", help="text to stylize")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="output all styles (default)")
    mode.add_argument("--style", help="output a single style by name")
    p.add_argument("--list-styles", action="store_true", help="list available styles")
    p.add_argument(
        "--labels", action="store_true", help="prefix each line with its style name"
    )
    p.add_argument("--gui", action="store_true", help="launch the Tkinter GUI")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.gui:
        return run_gui()
    if args.list_styles:
        for name in available_styles():
            print(name)
        return 0

    text = _get_input_text(args)
    if args.style:
        if args.style.strip().lower() == "all":
            print(transform_all(text, labels=args.labels))
            return 0
        style = normalize_style_name(args.style)
        if style not in STYLES:
            print(f"Unknown style: {args.style}", file=sys.stderr)
            print("Use --list-styles to see options.", file=sys.stderr)
            return 2
        print(transform(style, text))
        return 0

    print(transform_all(text, labels=args.labels))
    return 0


def _build_roman_numeral_style(lowercase: bool = False) -> Callable[[str], str]:
    def _style(text: str) -> str:
        return _transform_clusters(
            text,
            custom_digit_run=lambda run: _convert_digit_run_to_roman(
                run, lowercase=lowercase
            ),
        )

    return _style


def _safe_tk() -> None:
    try:
        import tkinter as tk  # type: ignore
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Tkinter is unavailable: {exc}") from exc


def run_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover
        print(f"Tkinter is unavailable: {exc}", file=sys.stderr)
        return 1

    class App:
        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title("Enfancy")
            self.root.geometry("1100x700")

            outer = ttk.Frame(self.root, padding=10)
            outer.pack(fill="both", expand=True)

            top = ttk.Frame(outer)
            top.pack(fill="x", pady=(0, 8))
            ttk.Label(top, text="Style").pack(side="left")
            self.style_var = tk.StringVar(value="all")
            self.style_box = ttk.Combobox(
                top,
                state="readonly",
                textvariable=self.style_var,
                values=["all"] + available_styles(),
                width=28,
            )
            self.style_box.pack(side="left", padx=8)
            self.style_box.set("all")
            ttk.Button(top, text="Convert", command=self.refresh).pack(side="left")

            panes = ttk.PanedWindow(outer, orient="horizontal")
            panes.pack(fill="both", expand=True)

            left_frame = ttk.Frame(panes)
            right_frame = ttk.Frame(panes)
            panes.add(left_frame, weight=1)
            panes.add(right_frame, weight=1)

            ttk.Label(left_frame, text="Input").pack(anchor="w")
            self.input_text = tk.Text(left_frame, wrap="word", undo=True)
            self.input_text.pack(side="left", fill="both", expand=True)
            left_scroll = ttk.Scrollbar(left_frame, command=self.input_text.yview)
            left_scroll.pack(side="right", fill="y")
            self.input_text.configure(yscrollcommand=left_scroll.set)
            self.input_text.insert("1.0", DEFAULT_SAMPLE)

            ttk.Label(right_frame, text="Output").pack(anchor="w")
            self.output_text = tk.Text(right_frame, wrap="word", state="disabled")
            self.output_text.pack(side="left", fill="both", expand=True)
            right_scroll = ttk.Scrollbar(right_frame, command=self.output_text.yview)
            right_scroll.pack(side="right", fill="y")
            self.output_text.configure(yscrollcommand=right_scroll.set)

            self._refresh_after = None
            self.input_text.bind("<KeyRelease>", self.schedule_refresh)
            self.style_box.bind("<<ComboboxSelected>>", self.schedule_refresh)
            self.refresh()

        def schedule_refresh(self, _event=None) -> None:
            if self._refresh_after is not None:
                self.root.after_cancel(self._refresh_after)
            self._refresh_after = self.root.after(120, self.refresh)

        def refresh(self) -> None:
            self._refresh_after = None
            text = self.input_text.get("1.0", "end-1c") or DEFAULT_SAMPLE
            selected = self.style_var.get().strip().lower()
            style = normalize_style_name(self.style_var.get())
            if selected == "all":
                result = transform_all(text)
            elif style in STYLES:
                result = transform(style, text)
            else:
                result = transform_all(text)
            self.output_text.configure(state="normal")
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", result)
            self.output_text.configure(state="disabled")

    App().root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
