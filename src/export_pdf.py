"""Render the Markdown report as a print-ready PDF.

Written rather than delegated because no Markdown-to-PDF converter is available
on this machine: pandoc, wkhtmltopdf and a LaTeX distribution are all absent,
and WeasyPrint needs GTK libraries that Windows does not supply. ReportLab is
pure Python and installs cleanly, so the conversion is done directly.

Two details decide whether the output looks professional or broken.

**Fonts must be registered from TrueType files.** ReportLab's built-in fonts are
WinAnsi-encoded and lack several characters this report uses -- the minus sign
U+2212, the arrow, the greater-or-equal sign and the Greek letters -- each of
which would render as a black box. The glyph coverage of every candidate font
was checked against the report's actual character set before choosing.

**Unicode subscripts and superscripts are translated, not passed through.**
Even in a font that contains them they are inconsistently sized and positioned,
and Arial omits three of the ones used here entirely. They are converted to
ReportLab's ``<sub>`` and ``<super>`` markup, which the text engine positions
properly.
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["render_pdf", "MarkdownParser"]

PAGE_FONTS = {
    "body": ("Times", "C:/Windows/Fonts/times.ttf"),
    "body-bold": ("Times-Bold", "C:/Windows/Fonts/timesbd.ttf"),
    "body-italic": ("Times-Italic", "C:/Windows/Fonts/timesi.ttf"),
    "body-bolditalic": ("Times-BoldItalic", "C:/Windows/Fonts/timesbi.ttf"),
    "mono": ("Mono", "C:/Windows/Fonts/consola.ttf"),
}

# Unicode sub/superscripts -> the digit, so they can be wrapped in markup.
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾", "0123456789+-=()")
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎", "0123456789+-=()")
SUPER_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾]+")
SUB_RUN = re.compile(r"[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎]+")


# --------------------------------------------------------------------------
# Inline formatting
# --------------------------------------------------------------------------


def inline(text: str) -> str:
    """Convert Markdown inline spans to ReportLab's mini-HTML.

    Escaping happens first so that literal ``<`` and ``&`` in the source cannot
    be read as markup, and the generated tags are added afterwards.
    """
    out = html.escape(text, quote=False)

    # Sub- and superscripts are translated before emphasis is parsed. In a span
    # like *K*<subscript one> the character after the closing asterisk is a
    # subscript digit, which \w matches, so the italic pattern failed there --
    # and its lazy quantifier then ran on to the next asterisk in the paragraph,
    # italicising an unrelated run and leaving stray asterisks on the page.
    out = SUPER_RUN.sub(lambda m: f"<super>{m.group(0).translate(SUPERSCRIPTS)}</super>", out)
    out = SUB_RUN.sub(lambda m: f"<sub>{m.group(0).translate(SUBSCRIPTS)}</sub>", out)

    # Code spans before emphasis, so underscores inside code survive.
    out = re.sub(r"`([^`]+)`", r'<font face="Mono" size="9">\1</font>', out)
    out = re.sub(r"\*\*\*([^*]+?)\*\*\*", r"<b><i>\1</i></b>", out)
    out = re.sub(r"\*\*([^*]+?)\*\*", r"<b>\1</b>", out)
    # [^*] rather than . so an unmatched asterisk cannot swallow the rest of the
    # paragraph looking for a partner.
    out = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", out)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1a4d80">\1</link>', out)


def plain(text: str) -> str:
    """Strip Markdown markers without adding markup, for headings and captions."""
    out = re.sub(r"`([^`]+)`", r"\1", text)
    out = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)
    return re.sub(r"\*(.+?)\*", r"\1", out)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


@dataclass
class Block:
    """One parsed Markdown block."""

    kind: str
    content: Any
    level: int = 0


class MarkdownParser:
    """Turn a Markdown document into an ordered list of blocks.

    Only the constructs this report uses are supported -- headings, paragraphs,
    pipe tables, images, fenced code, block quotes, ordered and unordered lists,
    and horizontal rules. A general Markdown implementation is not the goal.
    """

    def __init__(self, text: str) -> None:
        self.lines = text.replace("\r\n", "\n").split("\n")

    def parse(self) -> list[Block]:
        """Parse the document."""
        blocks: list[Block] = []
        index = 0
        while index < len(self.lines):
            line = self.lines[index]
            stripped = line.strip()

            if not stripped:
                index += 1
            elif stripped.startswith("```"):
                index = self._code(blocks, index)
            elif stripped.startswith("#"):
                index = self._heading(blocks, index)
            elif re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
                blocks.append(Block("rule", None))
                index += 1
            elif stripped.startswith("!["):
                index = self._image(blocks, index)
            elif stripped.startswith("|") and self._is_table(index):
                index = self._table(blocks, index)
            elif stripped.startswith(">"):
                index = self._quote(blocks, index)
            elif re.match(r"^(\d+\.|[-*+])\s", stripped):
                index = self._list(blocks, index)
            else:
                index = self._paragraph(blocks, index)
        return blocks

    # -- block readers --------------------------------------------------

    def _heading(self, blocks: list[Block], index: int) -> int:
        match = re.match(r"^(#{1,6})\s+(.*)$", self.lines[index].strip())
        assert match is not None
        blocks.append(Block("heading", plain(match.group(2)), len(match.group(1))))
        return index + 1

    def _code(self, blocks: list[Block], index: int) -> int:
        body: list[str] = []
        index += 1
        while index < len(self.lines) and not self.lines[index].strip().startswith("```"):
            body.append(self.lines[index])
            index += 1
        blocks.append(Block("code", "\n".join(body)))
        return index + 1

    def _image(self, blocks: list[Block], index: int) -> int:
        match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", self.lines[index].strip())
        if match is None:
            return self._paragraph(blocks, index)
        index += 1
        # A bolded "Figure N —" line directly beneath is this image's caption.
        caption: list[str] = []
        while index < len(self.lines) and not self.lines[index].strip():
            index += 1
        if index < len(self.lines) and self.lines[index].strip().startswith("**Figure"):
            while index < len(self.lines) and self.lines[index].strip():
                caption.append(self.lines[index].strip())
                index += 1
        blocks.append(Block("image", (match.group(2), " ".join(caption))))
        return index

    def _is_table(self, index: int) -> bool:
        following = self.lines[index + 1].strip() if index + 1 < len(self.lines) else ""
        return bool(re.match(r"^\|[\s:|-]+\|$", following))

    def _table(self, blocks: list[Block], index: int) -> int:
        def cells(row: str) -> list[str]:
            return [c.strip() for c in row.strip().strip("|").split("|")]

        header = cells(self.lines[index])
        aligns = []
        for spec in cells(self.lines[index + 1]):
            if spec.endswith(":") and spec.startswith(":"):
                aligns.append("CENTER")
            elif spec.endswith(":"):
                aligns.append("RIGHT")
            else:
                aligns.append("LEFT")
        index += 2

        rows = [header]
        while index < len(self.lines) and self.lines[index].strip().startswith("|"):
            rows.append(cells(self.lines[index]))
            index += 1

        # A bolded "Table N —" paragraph immediately above is this table's caption.
        caption = ""
        for previous in reversed(blocks[-2:]):
            if previous.kind == "paragraph" and previous.content.lstrip().startswith("**Table"):
                caption = previous.content
                blocks.remove(previous)
                break
        blocks.append(Block("table", (rows, aligns, caption)))
        return index

    def _quote(self, blocks: list[Block], index: int) -> int:
        body: list[str] = []
        while index < len(self.lines) and self.lines[index].strip().startswith(">"):
            body.append(self.lines[index].strip().lstrip(">").strip())
            index += 1
        blocks.append(Block("quote", " ".join(body)))
        return index

    def _list(self, blocks: list[Block], index: int) -> int:
        # The original marker is kept, not just whether the list is ordered, so a
        # numbered list renders with its numbers rather than a generic glyph.
        items: list[tuple[str, str]] = []
        while index < len(self.lines):
            stripped = self.lines[index].strip()
            match = re.match(r"^(\d+\.|[-*+])\s+(.*)$", stripped)
            if match:
                items.append((match.group(2), match.group(1)))
                index += 1
            elif stripped and items and not stripped.startswith(("|", "#", "!", ">", "```")):
                # A continuation line belongs to the item above it.
                items[-1] = (items[-1][0] + " " + stripped, items[-1][1])
                index += 1
            else:
                break
        blocks.append(Block("list", items))
        return index

    def _paragraph(self, blocks: list[Block], index: int) -> int:
        body: list[str] = []
        while index < len(self.lines):
            stripped = self.lines[index].strip()
            if not stripped or stripped.startswith(("#", "|", "!", ">", "```")):
                break
            if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
                break
            if re.match(r"^(\d+\.|[-*+])\s", stripped):
                break
            body.append(stripped)
            index += 1
        if body:
            blocks.append(Block("paragraph", " ".join(body)))
        return index


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _register_fonts() -> str:
    """Register the TrueType faces, returning the body font name.

    Raises:
        FileNotFoundError: If the fonts are not present on this system.
    """
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for key, (name, path) in PAGE_FONTS.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"font for {key!r} not found at {path}. Rendering with ReportLab's "
                "built-in fonts would turn several characters in this report into "
                "black boxes, so the conversion stops rather than producing that."
            )
        pdfmetrics.registerFont(TTFont(name, path))

    body = PAGE_FONTS["body"][0]
    addMapping(body, 0, 0, body)
    addMapping(body, 1, 0, PAGE_FONTS["body-bold"][0])
    addMapping(body, 0, 1, PAGE_FONTS["body-italic"][0])
    addMapping(body, 1, 1, PAGE_FONTS["body-bolditalic"][0])
    return body


def _styles(body: str) -> dict[str, Any]:
    """Paragraph styles for the document."""
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle

    ink = "#111111"
    return {
        "title": ParagraphStyle(
            "title",
            fontName=PAGE_FONTS["body-bold"][0],
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            textColor=ink,
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName=PAGE_FONTS["body-italic"][0],
            fontSize=12.5,
            leading=17,
            alignment=TA_CENTER,
            textColor="#444444",
            spaceAfter=8,
        ),
        "author": ParagraphStyle(
            "author",
            fontName=body,
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            textColor=ink,
            spaceBefore=18,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName=PAGE_FONTS["body-bold"][0],
            fontSize=16.5,
            leading=20,
            textColor=ink,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName=PAGE_FONTS["body-bold"][0],
            fontSize=13,
            leading=16.5,
            textColor=ink,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3",
            fontName=PAGE_FONTS["body-bolditalic"][0],
            fontSize=11.5,
            leading=15,
            textColor="#222222",
            spaceBefore=11,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName=body,
            fontSize=10.5,
            leading=15,
            alignment=TA_JUSTIFY,
            textColor=ink,
            spaceAfter=7,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName=body,
            fontSize=9,
            leading=12.5,
            alignment=TA_CENTER,
            textColor="#333333",
            spaceBefore=5,
            spaceAfter=12,
        ),
        "tablecaption": ParagraphStyle(
            "tablecaption",
            fontName=body,
            fontSize=9,
            leading=12.5,
            textColor="#333333",
            spaceBefore=8,
            spaceAfter=5,
        ),
        "cell": ParagraphStyle(
            "cell",
            fontName=body,
            fontSize=8.6,
            leading=11,
            textColor=ink,
        ),
        "cellhead": ParagraphStyle(
            "cellhead",
            fontName=PAGE_FONTS["body-bold"][0],
            fontSize=8.6,
            leading=11,
            textColor="#ffffff",
        ),
        "code": ParagraphStyle(
            "code",
            fontName=PAGE_FONTS["mono"][0],
            fontSize=8.2,
            leading=11.4,
            textColor="#1c1c1c",
            backColor="#f4f4f2",
            borderPadding=7,
            leftIndent=3,
            rightIndent=3,
            spaceBefore=5,
            spaceAfter=10,
        ),
        "quote": ParagraphStyle(
            "quote",
            fontName=PAGE_FONTS["body-italic"][0],
            fontSize=10.5,
            leading=15,
            leftIndent=16,
            rightIndent=10,
            textColor="#333333",
            borderPadding=4,
            spaceBefore=5,
            spaceAfter=9,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName=body,
            fontSize=10.5,
            leading=14.5,
            alignment=TA_JUSTIFY,
            textColor=ink,
            leftIndent=18,
            bulletIndent=4,
            spaceAfter=4,
            # Without these the marker is drawn in ReportLab's default Helvetica,
            # which mismatches the body typeface and leaves the bullet with no
            # usable Unicode mapping -- it extracts from the finished PDF as
            # U+007F rather than as a bullet.
            bulletFontName=body,
            bulletFontSize=10.5,
        ),
    }


def _table_flowable(block: Block, styles: dict[str, Any], width: float) -> list[Any]:
    """Build a table, sized to the available width."""
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    rows, aligns, caption = block.content
    columns = max(len(r) for r in rows)
    aligns = (aligns + ["LEFT"] * columns)[:columns]

    data = []
    for number, row in enumerate(rows):
        padded = (row + [""] * columns)[:columns]
        style = styles["cellhead"] if number == 0 else styles["cell"]
        data.append([Paragraph(inline(cell), style) for cell in padded])

    # Width by longest cell, so a column of numbers does not get the same space
    # as a column of prose, then scaled to fit the frame.
    weights = [
        max(len(plain(r[i])) if i < len(r) else 0 for r in rows) ** 0.62 for i in range(columns)
    ]
    total = sum(weights) or 1.0
    widths = [max(width * w / total, 26.0) for w in weights]
    if sum(widths) > width:
        scale = width / sum(widths)
        widths = [w * scale for w in widths]

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#33475b")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#33475b")),
        ("GRID", (0, 1), (-1, -1), 0.25, colors.HexColor("#c8ccd0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f7f8")]),
    ]
    for column, align in enumerate(aligns):
        commands.append(("ALIGN", (column, 0), (column, -1), align))

    table.setStyle(TableStyle(commands))
    flowables: list[Any] = []
    if caption:
        flowables.append(Paragraph(inline(caption), styles["tablecaption"]))
    flowables.append(table)
    return flowables


def _image_flowable(
    block: Block, styles: dict[str, Any], base: Path, width: float, dpi: int
) -> list[Any]:
    """Build a figure scaled to the frame, with its caption beneath.

    Figures are rendered at 300 dpi for print, which at this page width is about
    3,570 pixels across but is displayed at roughly 6.7 inches -- an effective
    530 dpi. Embedding that costs several megabytes per figure for resolution no
    printer will use, so each image is resampled to ``dpi`` at its displayed
    size before embedding.
    """
    from io import BytesIO

    from PIL import Image as PILImage
    from reportlab.platypus import Image, Paragraph, Spacer

    source, caption = block.content
    path = (base / source).resolve()
    if not path.exists():
        return [Paragraph(f"[missing figure: {source}]", styles["caption"])]

    with PILImage.open(path) as handle:
        native_width, native_height = handle.size
        scale = width / native_width
        height = native_height * scale
        # Keep a figure from consuming a whole page.
        max_height = 400.0
        if height > max_height:
            scale = max_height / native_height
            width, height = native_width * scale, max_height

        # Points are 1/72 inch, so the pixel budget is the displayed inches x dpi.
        wanted = max(1, int(round(width / 72.0 * dpi)))
        if wanted < native_width:
            ratio = wanted / native_width
            resized = handle.convert("RGB").resize(
                (wanted, max(1, int(round(native_height * ratio)))),
                PILImage.LANCZOS,
            )
        else:
            resized = handle.convert("RGB")

        # Quantise to a 256-colour palette. Resampling anti-aliases crisp line
        # art into many near-identical shades, which PNG compresses badly; a
        # palette removes them. Measured on one figure: 328 KB as RGB, 133 KB
        # palettised, against 173 KB as JPEG -- and JPEG rings around thin lines
        # and axis labels, which these figures are mostly made of. Dithering is
        # off because it adds noise that both enlarges the file and looks worse
        # on a smooth colour map.
        palettised = resized.quantize(
            colors=256, method=PILImage.MEDIANCUT, dither=PILImage.Dither.NONE
        )

        buffer = BytesIO()
        palettised.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

    # The buffer is passed directly: Image() treats an ImageReader as a path and
    # calls splitext on it.
    flowables: list[Any] = [Spacer(1, 4), Image(buffer, width=width, height=height)]
    if caption:
        flowables.append(Paragraph(inline(caption), styles["caption"]))
    return flowables


def _build_story(
    blocks: list[Block], styles: dict[str, Any], base: Path, width: float, dpi: int
) -> list[Any]:
    """Turn parsed blocks into ReportLab flowables."""
    from reportlab.platypus import HRFlowable, KeepTogether, PageBreak, Paragraph, Spacer

    story: list[Any] = []
    seen_first_heading = False

    for block in blocks:
        if block.kind == "heading":
            if block.level == 1:
                # Each top-level section starts a page, except the first.
                if seen_first_heading:
                    story.append(PageBreak())
                seen_first_heading = True
                story.append(Paragraph(inline(block.content), styles["h1"]))
                story.append(
                    HRFlowable(
                        width="100%", thickness=0.8, color="#33475b", spaceBefore=2, spaceAfter=10
                    )
                )
            else:
                key = {2: "h2", 3: "h3"}.get(block.level, "h3")
                story.append(Paragraph(inline(block.content), styles[key]))
        elif block.kind == "paragraph":
            story.append(Paragraph(inline(block.content), styles["body"]))
        elif block.kind == "table":
            story.extend(_table_flowable(block, styles, width))
            story.append(Spacer(1, 11))
        elif block.kind == "image":
            story.append(KeepTogether(_image_flowable(block, styles, base, width, dpi)))
        elif block.kind == "code":
            escaped = html.escape(block.content, quote=False).replace("\n", "<br/>")
            story.append(Paragraph(escaped or "&nbsp;", styles["code"]))
        elif block.kind == "quote":
            story.append(Paragraph(inline(block.content), styles["quote"]))
        elif block.kind == "list":
            for text, source_marker in block.content:
                # bulletText is drawn literally, not parsed as mini-HTML, so an
                # entity like &bull; would appear on the page as those characters.
                # A numbered list keeps its own numbering; anything else gets a
                # real bullet glyph, which the font was checked for.
                marker = source_marker if source_marker[0].isdigit() else "•"
                story.append(Paragraph(inline(text), styles["bullet"], bulletText=marker))
            story.append(Spacer(1, 5))
        elif block.kind == "rule":
            story.append(Spacer(1, 3))
    return story


def _title_page(text: str, styles: dict[str, Any]) -> tuple[list[Any], str]:
    """Build the title page, returning it and the running-header title."""
    from reportlab.platypus import HRFlowable, PageBreak, Paragraph, Spacer

    lines = text.split("\n")
    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), "Report")
    after = lines[lines.index(f"# {title}") + 1 :]
    meta = [ln.strip() for ln in after[:6] if ln.strip() and not ln.startswith("#")]
    meta = [m for m in meta if m != "---"]

    story: list[Any] = [
        Spacer(1, 150),
        Paragraph(inline(title), styles["title"]),
        HRFlowable(width="55%", thickness=0.8, color="#33475b", spaceBefore=6, spaceAfter=16),
    ]
    for line in meta[:2]:
        story.append(Paragraph(inline(line), styles["subtitle"]))
    story.append(PageBreak())
    return story, plain(title)


def render_pdf(source: Path, destination: Path, *, dpi: int = 300) -> Path:
    """Render a Markdown file to PDF.

    Args:
        source: The Markdown document.
        destination: Where to write the PDF.
        dpi: Effective resolution for embedded figures, at their displayed size.

    Returns:
        The path written.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

    body_font = _register_fonts()
    styles = _styles(body_font)
    text = source.read_text(encoding="utf-8")

    page_width, page_height = A4
    margin = 20 * mm
    frame_width = page_width - 2 * margin

    title_story, running = _title_page(text, styles)

    # Drop the title block; it is rendered as the title page instead.
    body_text = text
    first_h2 = body_text.find("\n## ")
    if first_h2 > 0:
        body_text = body_text[first_h2 + 1 :]

    blocks = MarkdownParser(body_text).parse()
    story = title_story + _build_story(blocks, styles, source.parent, frame_width, dpi)

    def decorate(canvas: Any, document: Any) -> None:
        canvas.saveState()
        if document.page > 1:
            canvas.setFont(body_font, 8)
            canvas.setFillColor("#666666")
            canvas.drawString(margin, page_height - margin + 6 * mm, running)
            canvas.drawRightString(
                page_width - margin, page_height - margin + 6 * mm, "Mahamatbt"
            )
            canvas.setStrokeColor("#cccccc")
            canvas.setLineWidth(0.4)
            canvas.line(
                margin,
                page_height - margin + 4 * mm,
                page_width - margin,
                page_height - margin + 4 * mm,
            )
            canvas.drawCentredString(page_width / 2, margin - 8 * mm, str(document.page))
        canvas.restoreState()

    document = BaseDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title=running,
        author="Mahamatbt",
    )
    frame = Frame(margin, margin, frame_width, page_height - 2 * margin, id="body")
    document.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.build(story)
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="report/REPORT.md", help="Markdown to render.")
    parser.add_argument("--out", default="report/REPORT.pdf", help="PDF to write.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Effective resolution for embedded figures. 300 is print quality.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render the report."""
    args = build_parser().parse_args(argv)
    path = render_pdf(Path(args.source), Path(args.out), dpi=args.dpi)
    size = path.stat().st_size
    print(f"wrote {path} ({size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
