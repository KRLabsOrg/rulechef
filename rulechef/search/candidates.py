"""Candidate regex generation from labelled positives, without an LLM.

Every candidate is a ``(pattern, text_template)`` pair. ``"$0"`` emits the
whole match, ``"$1"`` the first capture group (used by context rules).
Generation only reads training positives and their surrounding text; there
are no built-in lexicons, so every word alternation comes from the data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import regex

NUMBER = r"\d+(?:[.,]\d+)*"
_TOKEN = regex.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+|\s+|.", regex.U)
_LEFT_WORD = regex.compile(r"([^\W\d_]+\.?)\s*$")
_RIGHT_TOKEN = regex.compile(r"\s*(\S+?)(?:[\s,;.)]|$)")
_WORD_CLASS = {"WC": r"[A-Z][a-z]+", "WL": r"[a-z]+", "WU": r"[A-Z]{2,}", "WM": r"[A-Za-z]+"}
_MAX_ATOMS = 15
_MAX_VOCAB = 60

Atom = tuple[str, str]
CandidateKey = tuple[str, str]  # (pattern, text template)
Row = list[tuple[int, int, str]]  # (start, end, text) per match in one document


@dataclass
class Positive:
    """One gold span of the label being learned."""

    text: str  # stripped span text
    doc: int  # index of the document in the training texts
    start: int
    end: int


def atomise(text: str) -> list[Atom]:
    """Split a span into typed atoms: numbers, words by case shape, spaces, punctuation."""
    atoms: list[Atom] = []
    for tok in _TOKEN.findall(text):
        if tok[0].isdigit():
            atoms.append(("N", tok))
        elif tok.isspace():
            atoms.append(("S", tok))
        elif tok[0].isalpha():
            if tok[0].isupper() and (len(tok) == 1 or tok[1:].islower()):
                shape = "C"
            elif tok.isupper():
                shape = "U"
            elif tok.islower():
                shape = "L"
            else:
                shape = "M"
            atoms.append(("W" + shape, tok))
        else:
            atoms.append(("P", tok))
    return atoms


def signature(atoms: list[Atom]) -> tuple[str, ...]:
    """Atom kinds, with punctuation kept literal (``P/``, ``P-``)."""
    return tuple(kind if kind != "P" else "P" + value for kind, value in atoms)


def number_shape(token: str) -> str:
    """Exact digit-run shape of a number token, e.g. ``12,500`` -> ``\\d{2},\\d{3}``."""
    parts = regex.split(r"([.,])", token)
    return "".join(regex.escape(p) if p in ".," else rf"\d{{{len(p)}}}" for p in parts)


def alternation(words) -> str:
    """Longest-first alternation of the given literals."""
    unique = sorted(set(words), key=lambda w: (-len(w), w))
    if len(unique) == 1:
        return regex.escape(unique[0])
    return "(?:" + "|".join(regex.escape(w) for w in unique) + ")"


def build_group_regex(
    members: list[list[Atom]],
    level: tuple[str, str, str],
    vocab: dict[str, set[str]],
    boundaries: bool,
) -> str:
    """Generalise positives that share one atom signature into a regex.

    ``level`` is (number level, word level, space level):
    numbers ``exact`` / ``range`` / ``gen``; words ``lit`` (observed literals),
    ``vocab`` (label vocabulary, if small) or ``class``; spaces ``ws`` / ``sp``.
    """
    num_level, word_level, space_level = level
    parts = []
    for i, (kind, _) in enumerate(members[0]):
        values = [m[i][1] for m in members]
        if kind == "N":
            if num_level == "exact":
                shapes = sorted({number_shape(v) for v in values})
                parts.append(shapes[0] if len(shapes) == 1 else "(?:" + "|".join(shapes) + ")")
            elif num_level == "range" and all(v.isdigit() for v in values):
                lo, hi = min(map(len, values)), max(map(len, values))
                parts.append(rf"\d{{{lo}}}" if lo == hi else rf"\d{{{lo},{hi}}}")
            else:
                parts.append(NUMBER)
        elif kind.startswith("W"):
            if word_level == "lit":
                parts.append(alternation(values))
            elif word_level == "vocab":
                words = vocab.get(kind, set()) | set(values)
                parts.append(alternation(words) if len(words) <= _MAX_VOCAB else _WORD_CLASS[kind])
            else:
                parts.append(_WORD_CLASS[kind])
        elif kind == "S":
            parts.append(r"\s+" if space_level == "ws" else " ")
        else:
            parts.append(regex.escape(values[0]))
    pattern = "".join(parts)
    if boundaries:
        first, last = members[0][0][0], members[0][-1][0]
        if first == "N" or first.startswith("W"):
            pattern = r"\b" + pattern
        if last == "N" or last.startswith("W"):
            pattern = pattern + r"\b"
    return pattern


def group_candidates(positives: list[Positive], min_support: int) -> dict[CandidateKey, str]:
    """grex-style generalisation per atom signature, plus optional-block merges.

    Returns candidate -> origin. The origin names the signature group the
    candidate came from (``g3``) or the pair of groups it merges (``m3_1``).
    """
    groups: dict[tuple[str, ...], list[list[Atom]]] = defaultdict(list)
    for pos in positives:
        atoms = atomise(pos.text)
        if len(atoms) <= _MAX_ATOMS:
            groups[signature(atoms)].append(atoms)

    # Label vocabulary: words seen at least twice in positives, by case shape
    word_counts = Counter(
        (kind, value)
        for members in groups.values()
        for atoms in members
        for kind, value in atoms
        if kind.startswith("W")
    )
    vocab: dict[str, set[str]] = defaultdict(set)
    for (kind, value), n in word_counts.items():
        if n >= 2:
            vocab[kind].add(value)

    out: dict[CandidateKey, str] = {}

    def add(pattern: str, origin: str) -> None:
        out.setdefault((pattern, "$0"), origin)

    frequent = [(sig, members) for sig, members in groups.items() if len(members) >= min_support]
    for gi, (_, members) in enumerate(frequent):
        for num_level in ("exact", "range", "gen"):
            for word_level in ("lit", "vocab", "class"):
                for space_level in ("ws", "sp"):
                    for bnd in (True, False):
                        level = (num_level, word_level, space_level)
                        add(build_group_regex(members, level, vocab, bnd), f"g{gi}")

    # Optional-block merges: a signature that equals another one minus a
    # leading / trailing [atom, space] block becomes core with the block optional.
    by_sig = {sig: (gi, members) for gi, (sig, members) in enumerate(frequent)}
    levels = (("range", "vocab"), ("gen", "vocab"), ("range", "class"), ("gen", "class"))
    for sig, (gi, members) in by_sig.items():
        for cut in (2, 4):
            for side in ("pre", "suf"):
                sub = sig[cut:] if side == "pre" else sig[:-cut]
                if not sub or sub not in by_sig:
                    continue
                gj, sub_members = by_sig[sub]
                for num_level, word_level in levels:
                    level = (num_level, word_level, "ws")
                    core = build_group_regex(sub_members, level, vocab, False)
                    block_members = [m[:cut] if side == "pre" else m[-cut:] for m in members]
                    block = build_group_regex(block_members, level, vocab, False)
                    if side == "pre":
                        pattern = "(?:" + block + ")?" + core
                    else:
                        pattern = core + "(?:" + block + ")?"
                    add(r"\b" + pattern + r"\b", f"m{gi}_{gj}")
    return out


def context_candidates(
    texts: list[str],
    positives: list[Positive],
    cores: list[tuple[str, str]],
    top_words: int = 8,
) -> dict[CandidateKey, str]:
    """Left-context rules: ``\\bword\\s*(core)\\b`` with a ``$1`` template.

    The context words are the most frequent words right before positives
    (``no.``, ``Application``).
    """
    left = Counter()
    for pos in positives:
        window = texts[pos.doc][max(0, pos.start - 30) : pos.start]
        m = _LEFT_WORD.search(window)
        if m:
            left[m.group(1)] += 1
    out: dict[CandidateKey, str] = {}
    for word, _ in left.most_common(top_words):
        for core, origin in cores:
            pattern = r"\b" + regex.escape(word) + r"\s*(" + core.replace(r"\b", "") + r")\b"
            out[(pattern, "$1")] = "c" + origin
    return out


def mutation_candidates(cores: list[tuple[str, str]]) -> dict[CandidateKey, str]:
    """Single-step mutations: boundary toggle, quantifier widen / loosen, optional space."""
    out: dict[CandidateKey, str] = {}
    for pattern, origin in cores:
        variants = set()
        if r"\b" in pattern:
            variants.add(pattern.replace(r"\b", ""))
        else:
            variants.add(r"\b" + pattern + r"\b")
        variants.add(
            regex.sub(
                r"\\d\{(\d+),(\d+)\}",
                lambda m: rf"\d{{{max(1, int(m.group(1)) - 1)},{int(m.group(2)) + 1}}}",
                pattern,
            )
        )
        variants.add(regex.sub(r"\\d\{(\d+)\}", lambda m: rf"\d{{1,{int(m.group(1))}}}", pattern))
        variants.add(pattern.replace(r"\s+", r"\s?") if r"\s+" in pattern else pattern)
        for variant in variants:
            if variant != pattern:
                out[(variant, "$0")] = "u" + origin
    return out


def lookahead_candidates(
    ranked: list[tuple[CandidateKey, str, list[Row]]],
    texts: list[str],
    golds: list[Counter],
) -> dict[CandidateKey, str]:
    """Add ``(?!\\s*tok)`` for frequent right-context tokens of a candidate's FPs.

    ``ranked`` holds (candidate, origin, train rows) for the best candidates.
    A token is used only if it follows at least two false positives, never a
    true positive, and is at most 8 characters long.
    """
    out: dict[CandidateKey, str] = {}
    for (pattern, template), origin, rows in ranked:
        fp_right, tp_right = Counter(), Counter()
        for i, row in enumerate(rows):
            gold, used = golds[i], Counter()
            for _start, end, text in row:
                m = _RIGHT_TOKEN.match(texts[i][end : end + 12])
                token = m.group(1) if m else None
                key = text.strip()
                if used[key] < gold.get(key, 0):
                    used[key] += 1
                    tp_right[token] += 1
                else:
                    fp_right[token] += 1
        bad = [
            t
            for t, n in fp_right.most_common(5)
            if t and n >= 2 and tp_right[t] == 0 and len(t) <= 8
        ][:3]
        if bad and template == "$0":
            lookahead = "".join(r"(?!\s*" + regex.escape(t) + ")" for t in bad)
            out[(pattern + lookahead, "$0")] = "n" + origin
    return out
