"""Build Word and PDF versions of docs/project-guide.md.

The Markdown guide is the source of truth. This script:
  1. renders every ```mermaid block to a PNG (mermaid-cli, using the local Chrome),
  2. rewrites the Markdown for print (title block, diagrams as figures, no manual contents list),
  3. pandoc → .docx using a reference document with the house styles,
  4. pandoc → standalone HTML → Chrome headless → .pdf.

Usage:  uv run python docs/export/build.py
Needs:  pandoc, Node.js (npx), Google Chrome.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "project-guide.md"
OUT = HERE / "out"
DIAGRAMS = OUT / "diagrams"
BASENAME = "LLM-Inference-Performance-Lab-Guide"

TITLE = "LLM Inference Performance Lab"
SUBTITLE = "Project Guide: GPU and LLM inference performance engineering with production-ready serving"

CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
MERMAID_CLI = "@mermaid-js/mermaid-cli@11"
MERMAID_SCALE = 3
TEXT_WIDTH_IN = 6.77  # A4 minus 0.75in margins
MAX_FIGURE_HEIGHT_IN = 8.6

MERMAID_CONFIG = {
    "theme": "base",
    "themeVariables": {
        "fontFamily": "Helvetica, Arial, sans-serif",
        "fontSize": "15px",
        "primaryColor": "#eef2ff",
        "primaryBorderColor": "#6366f1",
        "primaryTextColor": "#1e1b4b",
        "lineColor": "#64748b",
        "clusterBkg": "#f8fafc",
        "clusterBorder": "#cbd5e1",
        "actorBkg": "#eef2ff",
        "actorBorder": "#6366f1",
        "noteBkgColor": "#fef9c3",
    },
    "flowchart": {"htmlLabels": True, "curve": "basis", "padding": 12},
}


def run(cmd: list[str], **kw) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)
    if result.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    return struct.unpack(">II", header[16:24])


# ---------------------------------------------------------------- diagrams


USED_DIAGRAMS: set[Path] = set()


def render_mermaid(code: str, tmp: Path) -> Path:
    digest = hashlib.sha256((code + json.dumps(MERMAID_CONFIG)).encode()).hexdigest()[:12]
    png = DIAGRAMS / f"{digest}.png"
    USED_DIAGRAMS.add(png)
    if png.exists():
        return png
    src = tmp / f"{digest}.mmd"
    src.write_text(code)
    run(
        [
            "npx", "-y", MERMAID_CLI,
            "-i", str(src), "-o", str(png),
            "-c", str(tmp / "mermaid.json"), "-p", str(tmp / "puppeteer.json"),
            "-b", "white", "-s", str(MERMAID_SCALE),
        ],
        env=os.environ | {"PUPPETEER_SKIP_DOWNLOAD": "1"},
    )
    return png


def figure_size_attr(png: Path) -> str:
    w_px, h_px = png_size(png)
    # mermaid-cli renders at MERMAID_SCALE × CSS pixels; 96 CSS px per inch.
    w_in = w_px / MERMAID_SCALE / 96
    h_in = h_px / MERMAID_SCALE / 96
    scale = min(1.0, TEXT_WIDTH_IN / w_in, MAX_FIGURE_HEIGHT_IN / h_in)
    return f"width={w_in * scale:.2f}in"


# ---------------------------------------------------------------- markdown


def prepare_markdown(md: str, tmp: Path) -> str:
    lines = md.splitlines()
    # Drop the H1: the title block replaces it.
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    md = "\n".join(lines)

    # Drop the manual contents list: Word and PDF get a generated table of contents.
    md = re.sub(r"\n## Contents\n.*?\n---\n", "\n", md, flags=re.DOTALL)

    # Relative links to other repo files don't resolve outside the repo: keep the text only.
    md = re.sub(r"\[([^\]]+)\]\((?!#|https?:)[^)]+\)", r"\1", md)

    heading = "Diagram"
    out: list[str] = []
    parts = re.split(r"(^#{2,4} [^\n]*$|```mermaid\n.*?```)", md, flags=re.MULTILINE | re.DOTALL)
    for part in parts:
        if re.match(r"^#{2,4} ", part):
            heading = part.lstrip("#").strip()
            out.append(part)
        elif part.startswith("```mermaid"):
            code = part[len("```mermaid\n"):-3]
            png = render_mermaid(code, tmp)
            out.append(f"![{heading}]({png}){{{figure_size_attr(png)}}}")
        else:
            out.append(part)

    front = (
        "---\n"
        f"title: \"{TITLE}\"\n"
        f"subtitle: \"{SUBTITLE}\"\n"
        f"date: \"Generated {datetime.now().astimezone().date().isoformat()} from docs/project-guide.md\"\n"
        "---\n"
    )
    return front + "".join(out)


# ---------------------------------------------------------------- docx styles


TABLE_STYLE = """<w:style w:type="table" w:default="1" w:styleId="Table">
    <w:name w:val="Table"/><w:basedOn w:val="TableNormal"/><w:qFormat/>
    <w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr>
    <w:rPr><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr>
    <w:tblPr>
      <w:tblInd w:w="0" w:type="dxa"/>
      <w:tblBorders>
        <w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
        <w:left w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
        <w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
        <w:insideV w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
      </w:tblBorders>
      <w:tblCellMar>
        <w:top w:w="50" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>
        <w:bottom w:w="50" w:type="dxa"/><w:right w:w="90" w:type="dxa"/>
      </w:tblCellMar>
    </w:tblPr>
    <w:tblStylePr w:type="firstRow">
      <w:rPr><w:b/><w:color w:val="1E293B"/></w:rPr>
      <w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="E2E8F0"/><w:vAlign w:val="bottom"/></w:tcPr>
    </w:tblStylePr>
  </w:style>"""

SOURCE_CODE_STYLE = """<w:style w:type="paragraph" w:customStyle="1" w:styleId="SourceCode">
    <w:name w:val="Source Code"/><w:basedOn w:val="Normal"/><w:link w:val="VerbatimChar"/>
    <w:pPr>
      <w:pBdr><w:left w:val="single" w:sz="18" w:space="6" w:color="94A3B8"/></w:pBdr>
      <w:shd w:val="clear" w:color="auto" w:fill="F1F5F9"/>
      <w:wordWrap w:val="off"/>
      <w:spacing w:before="60" w:after="160" w:line="240" w:lineRule="auto"/>
      <w:ind w:left="120"/>
    </w:pPr>
    <w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>
  </w:style>"""


def build_reference_docx(path: Path) -> None:
    run(["pandoc", "-o", str(path), "--print-default-data-file", "reference.docx"])
    with zipfile.ZipFile(path) as z:
        files = {name: z.read(name) for name in z.namelist()}

    styles = files["word/styles.xml"].decode()
    font = '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Calibri" w:cs="Calibri"/>'
    styles = re.sub(r'<w:rFonts w:asciiTheme="m(?:ajor|inor)HAnsi"[^>]*/>', font, styles)
    # Body text 10.5pt, a little tighter than pandoc's default.
    styles = styles.replace('<w:sz w:val="24" />\n        <w:szCs w:val="24" />',
                            '<w:sz w:val="21" />\n        <w:szCs w:val="21" />', 1)
    styles = styles.replace('w:val="0F4761"', 'w:val="1E3A8A"')
    styles = re.sub(r'<w:style w:type="table" w:default="1" w:styleId="Table">.*?</w:style>',
                    TABLE_STYLE, styles, flags=re.DOTALL)
    # Table cell paragraphs ("Compact"): no spacing, small text.
    styles = re.sub(
        r'(<w:style [^>]*w:styleId="Compact">.*?<w:pPr>).*?(</w:pPr>)',
        r'\1<w:spacing w:before="20" w:after="20"/>\2<w:rPr><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr>',
        styles, count=1, flags=re.DOTALL,
    )
    styles = re.sub(
        r'(<w:style [^>]*w:styleId="VerbatimChar">.*?<w:rPr>).*?(</w:rPr>)',
        r'\1<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>'
        r'<w:color w:val="9F1239"/><w:sz w:val="18"/><w:szCs w:val="18"/>\2',
        styles, count=1, flags=re.DOTALL,
    )
    styles = styles.replace("</w:styles>", SOURCE_CODE_STYLE + "\n</w:styles>")
    files["word/styles.xml"] = styles.encode()

    # A4 with 0.75in margins.
    document = files["word/document.xml"].decode()
    document = re.sub(r"<w:pgSz[^>]*/>", '<w:pgSz w:w="11906" w:h="16838"/>', document)
    document = re.sub(
        r"<w:pgMar[^>]*/>",
        '<w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" '
        'w:header="567" w:footer="567" w:gutter="0"/>',
        document,
    )
    files["word/document.xml"] = document.encode()

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)


def finalize_docx(path: Path) -> None:
    """Full-width tables that autofit to content, and a default content type for PNGs."""
    with zipfile.ZipFile(path) as z:
        files = {name: z.read(name) for name in z.namelist()}
    document = files["word/document.xml"].decode()
    document = document.replace('<w:tblW w:type="auto" w:w="0" />', '<w:tblW w:type="pct" w:w="5000" />')
    files["word/document.xml"] = document.encode()
    types = files["[Content_Types].xml"].decode()
    if 'Extension="png"' not in types:
        types = types.replace("<Default ", '<Default Extension="png" ContentType="image/png" /><Default ', 1)
    files["[Content_Types].xml"] = types.encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", files.pop("[Content_Types].xml"))
        for name, data in files.items():
            z.writestr(name, data)


# ---------------------------------------------------------------- main


PANDOC_COMMON = [
    "--from", "markdown+gfm_auto_identifiers+emoji-tex_math_dollars",
    "--shift-heading-level-by=-1",
    "--toc", "--toc-depth=2",
    "--columns=100000",  # keep pipe tables auto-width instead of equal-width columns
    "--syntax-highlighting=none",
]


def main() -> None:
    for tool in ("pandoc", "npx"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")
    if not Path(CHROME).exists():
        sys.exit(f"Chrome not found at {CHROME} (set CHROME=...)")

    DIAGRAMS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "mermaid.json").write_text(json.dumps(MERMAID_CONFIG))
        (tmp / "puppeteer.json").write_text(
            json.dumps({"executablePath": CHROME, "args": ["--no-sandbox"]})
        )

        md = tmp / "guide.md"
        md.write_text(prepare_markdown(SOURCE.read_text(), tmp))

        reference = tmp / "reference.docx"
        build_reference_docx(reference)
        docx = OUT / f"{BASENAME}.docx"
        run(["pandoc", str(md), "-o", str(docx), f"--reference-doc={reference}", *PANDOC_COMMON])
        finalize_docx(docx)

        html = tmp / "guide.html"
        run([
            "pandoc", str(md), "-o", str(html), "--standalone", "--embed-resources",
            f"--css={HERE / 'print.css'}", "--metadata", f"pagetitle={TITLE}", *PANDOC_COMMON,
        ])
        pdf = OUT / f"{BASENAME}.pdf"
        run([
            CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}", html.as_uri(),
        ])

    # Drop cached diagrams that the guide no longer uses.
    for png in DIAGRAMS.glob("*.png"):
        if png not in USED_DIAGRAMS:
            png.unlink()

    print(f"wrote {docx.relative_to(Path.cwd()) if docx.is_relative_to(Path.cwd()) else docx}")
    print(f"wrote {pdf.relative_to(Path.cwd()) if pdf.is_relative_to(Path.cwd()) else pdf}")


if __name__ == "__main__":
    main()
