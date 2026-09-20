"""Tests for the Markdown-to-PDF renderer.

The renderer is bespoke because no converter was available on this machine, so
the parsing it does is worth pinning: a table read as paragraphs, or a caption
detached from its figure, produces a plausible-looking document that is quietly
wrong. The glyph checks matter for a different reason -- ReportLab renders a
character its font lacks as a solid black box rather than raising, so a missing
glyph is invisible to everything except a reader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import PROJECT_ROOT
from src.export_pdf import PAGE_FONTS, MarkdownParser, inline, plain

REPORT = PROJECT_ROOT / "report" / "REPORT.md"
PDF = PROJECT_ROOT / "report" / "REPORT.pdf"


# --------------------------------------------------------------------------
# Inline formatting
# --------------------------------------------------------------------------


def test_bold_italic_and_code_become_markup() -> None:
    assert inline("**bold**") == "<b>bold</b>"
    assert inline("*em*") == "<i>em</i>"
    assert '<font face="Mono"' in inline("`code`")


def test_markup_characters_in_the_source_are_escaped() -> None:
    """A literal < in the text must not be read as a tag."""
    assert "&lt;" in inline("a < b")
    assert "&amp;" in inline("Q & A")


def test_underscores_inside_code_spans_survive() -> None:
    assert "best_epoch" in inline("`best_epoch`")


def test_superscripts_become_super_tags() -> None:
    """Passed through literally these are missing glyphs in several fonts."""
    rendered = inline("4.3 × 10⁻⁵")
    assert "<super>-5</super>" in rendered
    assert "⁻" not in rendered and "⁵" not in rendered


def test_subscripts_become_sub_tags() -> None:
    rendered = inline("K₁ and K₂")
    assert "<sub>1</sub>" in rendered
    assert "<sub>2</sub>" in rendered
    assert "₁" not in rendered


def test_plain_strips_markers_without_adding_markup() -> None:
    assert plain("**Table 9 — Selected**") == "Table 9 — Selected"
    assert "<" not in plain("*italic* and `code`")


# --------------------------------------------------------------------------
# Block parsing
# --------------------------------------------------------------------------


def _kinds(markdown: str) -> list[str]:
    return [b.kind for b in MarkdownParser(markdown).parse()]


def test_headings_carry_their_level() -> None:
    blocks = MarkdownParser("# One\n\n## Two\n\n### Three\n").parse()
    assert [(b.kind, b.level) for b in blocks] == [
        ("heading", 1),
        ("heading", 2),
        ("heading", 3),
    ]


def test_pipe_table_is_parsed_with_alignment() -> None:
    markdown = "| a | b |\n|---|---:|\n| 1 | 2 |\n| 3 | 4 |\n"
    blocks = MarkdownParser(markdown).parse()
    assert len(blocks) == 1 and blocks[0].kind == "table"
    rows, aligns, _ = blocks[0].content
    assert rows == [["a", "b"], ["1", "2"], ["3", "4"]]
    assert aligns == ["LEFT", "RIGHT"]


def test_a_table_caption_above_it_is_attached_not_left_as_prose() -> None:
    """Otherwise the caption floats away from its table across a page break."""
    markdown = "**Table 3 — Things.**\n\n| a |\n|---|\n| 1 |\n"
    blocks = MarkdownParser(markdown).parse()
    assert [b.kind for b in blocks] == ["table"]
    assert "Table 3" in blocks[0].content[2]


def test_a_figure_caption_below_it_is_attached() -> None:
    markdown = "![alt](figures/x.png)\n\n**Figure 4 —** What it shows.\n"
    blocks = MarkdownParser(markdown).parse()
    assert [b.kind for b in blocks] == ["image"]
    source, caption = blocks[0].content
    assert source == "figures/x.png"
    assert caption.startswith("**Figure 4")


def test_a_paragraph_after_an_image_is_not_swallowed_as_a_caption() -> None:
    markdown = "![alt](figures/x.png)\n\nOrdinary prose follows.\n"
    assert _kinds(markdown) == ["image", "paragraph"]


def test_fenced_code_is_kept_verbatim() -> None:
    markdown = "```bash\npython run.py test\n# a comment\n```\n"
    blocks = MarkdownParser(markdown).parse()
    assert blocks[0].kind == "code"
    assert blocks[0].content == "python run.py test\n# a comment"


def test_lists_keep_their_original_markers() -> None:
    """The marker is kept, not just orderedness, so numbers survive to the page."""
    blocks = MarkdownParser("- one\n- two\n").parse()
    assert blocks[0].kind == "list"
    assert [marker for _, marker in blocks[0].content] == ["-", "-"]

    blocks = MarkdownParser("1. one\n2. two\n").parse()
    assert [marker for _, marker in blocks[0].content] == ["1.", "2."]


def test_wrapped_list_items_stay_with_their_item() -> None:
    blocks = MarkdownParser("- a claim that runs\n  onto a second line\n- next\n").parse()
    items = blocks[0].content
    assert len(items) == 2
    assert "onto a second line" in items[0][0]


def test_horizontal_rules_are_not_read_as_a_table_divider() -> None:
    assert _kinds("text\n\n---\n\nmore\n") == ["paragraph", "rule", "paragraph"]


def test_block_quotes_are_parsed() -> None:
    assert _kinds("> a quoted line\n> continued\n") == ["quote"]


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------


def test_every_character_in_the_report_has_a_glyph() -> None:
    """A missing glyph renders as a black box and raises nothing.

    Checked against the font's actual cmap rather than by eye, because the
    failure is silent and only visible in the rendered document.
    """
    if not REPORT.exists():
        pytest.skip("REPORT.md not present")
    fonttools = pytest.importorskip("fontTools.ttLib")

    text = REPORT.read_text(encoding="utf-8")
    # Sub/superscripts are translated to markup before rendering, so exclude them.
    translated = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
    needed = {ch for ch in text if ord(ch) > 127} - translated

    path = Path(PAGE_FONTS["body"][1])
    if not path.exists():
        pytest.skip(f"{path} not present on this system")

    font = fonttools.TTFont(str(path), fontNumber=0)
    covered: set[int] = set()
    for table in font["cmap"].tables:
        covered |= set(table.cmap.keys())

    missing = sorted(f"U+{ord(c):04X} {c!r}" for c in needed if ord(c) not in covered)
    assert missing == [], f"{path.name} lacks glyphs for: {missing}"


# --------------------------------------------------------------------------
# The rendered document
# --------------------------------------------------------------------------


def test_rendered_pdf_is_complete() -> None:
    """Checks the built artefact rather than rebuilding it, which is slow."""
    if not PDF.exists():
        pytest.skip("REPORT.pdf not built; run `python run.py pdf`")
    pypdf = pytest.importorskip("pypdf")

    reader = pypdf.PdfReader(str(PDF))
    assert len(reader.pages) > 20, "the report is shorter than expected"

    text = "".join(page.extract_text() or "" for page in reader.pages)
    assert "missing figure" not in text, "a figure reference did not resolve"

    images = sum(len(page.images) for page in reader.pages)
    figures = REPORT.read_text(encoding="utf-8").count("![")
    assert images == figures, f"{figures} figures in source, {images} embedded"

    for probe in ("Abstract", "References", "Appendix C", "MASE"):
        assert probe in text, f"{probe!r} missing from the rendered PDF"


# --------------------------------------------------------------------------
# Markup must not reach the page
# --------------------------------------------------------------------------


def test_italics_survive_a_following_subscript() -> None:
    r"""Regression: *K* followed by a subscript broke the italic pattern.

    A subscript digit matches \w, so the lookahead after the closing asterisk
    failed, and the lazy quantifier then ran on to the next asterisk in the
    paragraph -- italicising an unrelated run and leaving stray asterisks on the
    rendered page.
    """
    assert inline("*K*₁ = 2") == "<i>K</i><sub>1</sub> = 2"
    assert inline("2·*K*₂ weekly") == "2·<i>K</i><sub>2</sub> weekly"


def test_an_unmatched_asterisk_cannot_swallow_the_paragraph() -> None:
    rendered = inline("a * b and then *real emphasis* here")
    assert "<i>real emphasis</i>" in rendered
    assert rendered.count("<i>") == 1


def test_list_markers_are_characters_not_entities() -> None:
    """bulletText is drawn literally, so an entity appears as its own text."""
    blocks = MarkdownParser("- one\n- two\n").parse()
    markers = [marker for _, marker in blocks[0].content]
    assert markers == ["-", "-"]

    blocks = MarkdownParser("1. first\n2. second\n").parse()
    assert [m for _, m in blocks[0].content] == ["1.", "2."]


def test_rendered_pdf_contains_no_markup_artefacts() -> None:
    """The check that would have caught &bull; and &ndash; on the page."""
    if not PDF.exists():
        pytest.skip("REPORT.pdf not built; run `python run.py pdf`")
    pypdf = pytest.importorskip("pypdf")
    import re as _re

    text = "".join(page.extract_text() or "" for page in pypdf.PdfReader(str(PDF)).pages)

    offenders = {
        "HTML entities": _re.findall(r"&[a-zA-Z]+;?", text),
        "raw HTML tags": _re.findall(r"</?(?:b|i|font|sub|super|link|br)\b[^>]*>", text),
        "stray asterisks": _re.findall(r"\*", text),
        "markdown headings": _re.findall(r"(?m)^#{1,6}\s", text),
        "table pipe rules": _re.findall(r"\|\s*-{2,}", text),
        "image or link syntax": _re.findall(r"!\[|\]\(", text),
        "backticks": _re.findall(r"`", text),
        "control characters": [c for c in text if ord(c) < 32 and c not in "\n\r\t"],
        "replacement characters": _re.findall("�", text),
    }
    found = {name: len(hits) for name, hits in offenders.items() if hits}
    assert found == {}, f"markup reached the rendered page: {found}"


def test_rendered_pdf_uses_real_bullet_glyphs() -> None:
    if not PDF.exists():
        pytest.skip("REPORT.pdf not built")
    pypdf = pytest.importorskip("pypdf")
    text = "".join(page.extract_text() or "" for page in pypdf.PdfReader(str(PDF)).pages)
    assert "•" in text, "unordered lists have no bullet glyph"
