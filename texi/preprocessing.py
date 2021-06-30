# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Callable
from typing import Optional

LIGATURE_TABLE = {
    42802: "AA",
    42803: "aa",
    198: "AE",
    230: "ae",
    42804: "AO",
    42805: "ao",
    42806: "AU",
    42807: "au",
    42808: "AV",
    42809: "av",
    42810: "AV",
    42811: "av",
    42812: "AY",
    42813: "ay",
    128624: "et",
    64256: "ff",
    64259: "fi",
    64260: "fl",
    64257: "fi",
    64258: "fl",
    338: "OE",
    339: "oe",
    42830: "OO",
    42831: "oo",
    7838: "ſs",
    223: "ſz",
    64262: "st",
    64261: "ſt",
    42792: "TZ",
    42793: "tz",
    7531: "ue",
    42848: "VY",
    42849: "vy",
}


def split(text: str, sep: str) -> list:
    """Split text by separators."""
    sep = re.escape("".join(sep))

    return re.findall(rf"[^{sep}]+[{sep}]?", text)


def remove_control_chars(text: str) -> str:
    """Remove all control characters.

    Be careful that '\n' is will be removed. If you would like to split
    text later, just be careful.
    """

    return "".join(c for c in text if unicodedata.category(c)[0] != "C")


def remove_english_punctuations(text: str) -> str:
    """Remove all English punctuations."""

    return text.translate(str.maketrans("", "", string.punctuation))


def replace_whitesplaces(text: str, replacement: str = " ") -> str:
    """Replace consecutive whitespaces."""

    return re.sub(r"\s+", replacement, text)


def replace_ligatures(text: str) -> str:
    """Replace ligatures with non-ligatures."""

    return text.translate(LIGATURE_TABLE)


def get_opencc(conversion: Optional[str] = "t2s") -> Callable[[str], str]:
    # pylint: disable=import-outside-toplevel
    import opencc

    converter = opencc.OpenCC(conversion)

    def _wrapper(text):
        return converter.convert(text)

    return _wrapper
