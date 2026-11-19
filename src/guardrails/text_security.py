"""Bounded text views for Blue's filters. Never execute decoded user content."""
from __future__ import annotations

import base64
import binascii
import codecs
import html
import re
import unicodedata
from urllib.parse import unquote


MAX_INPUT_CHARS = 16_000
MAX_OUTPUT_CHARS = 64_000
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
    "у": "y", "і": "i", "ј": "j", "ѕ": "s", "α": "a", "ο": "o",
    "ρ": "p", "ι": "i", "κ": "k", "τ": "t",
})
_ESCAPE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|x([0-9a-fA-F]{2}))")
_TOKEN = re.compile(r"(?<![\w+/-])[A-Za-z0-9+/_-]{8,4096}={0,2}(?![\w+/-])")


class InspectionLimitError(ValueError):
    """Too many decoded views: refuse rather than inspect a truncated subset."""


def normalize_text(text: str) -> str:
    """NFKC and invisible controls; retain accents, case and visible layout."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return "".join(
        " " if unicodedata.category(c) == "Cc" and c not in "\n\r\t" else c
        for c in normalized
        if unicodedata.category(c) != "Cf" and c != "\u034f"
        and not (0xFE00 <= ord(c) <= 0xFE0F or 0xE0100 <= ord(c) <= 0xE01EF)
    )


def policy_text(text: str) -> str:
    """A matching view only: fold Vietnamese accents and common homoglyphs."""
    text = normalize_text(text).casefold().translate(_CONFUSABLES).replace("đ", "d")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def _unescape(match: re.Match) -> str:
    value = int(next(v for v in match.groups() if v is not None), 16)
    return chr(value) if value <= 0x10FFFF else match.group()


def security_views(text: str) -> list[str]:
    """At most 16 distinct views and two decode rounds.

    This detects common encodings, not arbitrary programs or every possible codec.
    The original request is still what the model sees if input checks allow it.
    """
    views = [normalize_text(text)]
    frontier = views[:]
    for _ in range(2):
        following = []
        for view in frontier:
            candidates = [normalize_text(_ESCAPE.sub(_unescape, unquote(html.unescape(view))))]
            # Decimal character codes and acrostics are transformations too.
            for match in re.finditer(r"(?<!\d)(?:\d{2,3}\s*[,;]\s*){2,127}\d{2,3}(?!\d)", view):
                decoded = "".join(chr(int(n)) for n in re.findall(r"\d+", match.group()))
                if all(c.isprintable() for c in decoded):
                    candidates.append(decoded)
            lines = [line for line in view.splitlines() if line.strip()]
            if 3 <= len(lines) <= 128:
                initials = [re.match(r"\s*(?:[-*]\s+)?(\S)(?:\s|[).:-]|$)", line) for line in lines]
                if all(initials):
                    candidates.append("".join(match.group(1) for match in initials))
            for match in _TOKEN.finditer(view):
                token = match.group()
                try:
                    decoded = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True).decode("utf-8")
                    if decoded and all(c.isprintable() or c in "\n\r\t" for c in decoded):
                        candidates.append(normalize_text(decoded))
                except (ValueError, UnicodeError, binascii.Error):
                    pass
                if len(token) % 2 == 0 and re.fullmatch(r"[0-9a-fA-F]+", token):
                    try:
                        decoded = bytes.fromhex(token).decode("utf-8")
                        if decoded and all(c.isprintable() or c in "\n\r\t" for c in decoded):
                            candidates.append(normalize_text(decoded))
                    except (ValueError, UnicodeError):
                        pass
            rot = re.search(r"\brot13\s*[:=(]\s*(.{1,4096})", view, re.IGNORECASE)
            if rot:
                candidates.append(codecs.decode(rot.group(1), "rot_13"))
            for candidate in candidates:
                if candidate and candidate not in views:
                    views.append(candidate)
                    following.append(candidate)
                    if len(views) >= 16:
                        raise InspectionLimitError("Too many decoded text views")
        frontier = following
    return views
