"""
Post-process LLM text so it sounds natural when spoken by Alexa.

Pipeline (order matters):
  1. Expand abbreviations
  2. Strip code blocks (unreadable when spoken)
  3. Convert markdown links to just the label
  4. Convert numbered lists  →  "First, ... Second, ..."
  5. Convert bullet lists    →  items joined with ". "
  6. Strip remaining markdown symbols
  7. Collapse whitespace
"""

import re

_ORDINALS = [
    "", "First", "Second", "Third", "Fourth", "Fifth",
    "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
]

_ABBREVIATIONS = [
    (r"\be\.g\.",        "for example"),
    (r"\bi\.e\.",        "that is"),
    (r"\betc\.",         "and so on"),
    (r"\bvs\.",          "versus"),
    (r"\bDr\.",          "Doctor"),
    (r"\bMr\.",          "Mister"),
    (r"\bMrs\.",         "Missus"),
    (r"\bMs\.",          "Miss"),
    (r"\bProf\.",        "Professor"),
    (r"\bapprox\.",      "approximately"),
    (r"\best\.",         "estimated"),
    (r"\bmin\.",         "minutes"),
    (r"\bmax\.",         "maximum"),
    (r"\bno\.",          "number"),
]


def _expand_abbreviations(text):
    for pattern, replacement in _ABBREVIATIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _strip_code_blocks(text):
    # Fenced code blocks: ```...``` (possibly with language tag)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Inline code: `...`
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def _convert_links(text):
    # [label](url) → label
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)


def _convert_numbered_list(text):
    """
    Turn:
        1. Do this
        2. Do that
        3. And this
    Into:
        First, Do this. Second, Do that. Third, And this.
    """
    lines = text.splitlines()
    result = []
    list_items = []

    for line in lines:
        m = re.match(r"^\s*(\d+)\.\s+(.+)", line)
        if m:
            n = int(m.group(1))
            label = _ORDINALS[n] if n < len(_ORDINALS) else f"Item {n}"
            list_items.append(f"{label}, {m.group(2).strip()}")
        else:
            if list_items:
                result.append(". ".join(list_items) + ".")
                list_items = []
            result.append(line)

    if list_items:
        result.append(". ".join(list_items) + ".")

    return "\n".join(result)


def _convert_bullet_list(text):
    """
    Turn:
        - item one
        - item two
        * item three
    Into:
        item one. item two. item three.
    """
    lines = text.splitlines()
    result = []
    list_items = []

    for line in lines:
        m = re.match(r"^\s*[-*•]\s+(.+)", line)
        if m:
            list_items.append(m.group(1).strip())
        else:
            if list_items:
                result.append(". ".join(list_items) + ".")
                list_items = []
            result.append(line)

    if list_items:
        result.append(". ".join(list_items) + ".")

    return "\n".join(result)


def _strip_markdown_symbols(text):
    # Headers: ## Heading → Heading
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold + italic: ***text*** or ___text___
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)
    # Bold: **text** or __text__
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
    # Italic: *text* or _text_
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    # Horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Blockquotes
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return text


def _collapse_whitespace(text):
    # Multiple blank lines → single blank line, then collapse to single space
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


_CHUNK_CHAR_LIMIT = 750


def chunk_text(text, limit=_CHUNK_CHAR_LIMIT):
    """
    Split text into chunks of at most `limit` characters, breaking on sentence
    boundaries where possible so each chunk sounds natural when spoken.
    Returns a list of strings. If the text fits in one chunk, returns a
    single-element list.
    """
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > limit:
        # Try to break at a sentence boundary within the limit
        window = remaining[:limit]
        cut = max(
            window.rfind(". "),
            window.rfind("? "),
            window.rfind("! "),
        )
        if cut != -1:
            cut += 1  # include the punctuation, exclude the space
        else:
            # Fall back to the last word boundary
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit  # hard cut if no boundary found

        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def to_voice(text):
    """Convert LLM output to clean, natural-sounding speech for Alexa."""
    if not text:
        return text

    text = _expand_abbreviations(text)
    text = _strip_code_blocks(text)
    text = _convert_links(text)
    text = _convert_numbered_list(text)
    text = _convert_bullet_list(text)
    text = _strip_markdown_symbols(text)
    text = _collapse_whitespace(text)

    return text
