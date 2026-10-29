#!/usr/bin/env python3
"""Build a landscape LaTeX/PDF catalogue from CUDAHeat YAML enrichment rules.

Usage:
    python generate_rules_catalogue.py

Outputs:
    generate_rules_catalogue.tex
    generate_rules_catalogue_table.tex
    generate_rules_catalogue.pdf

Requires:
    PyYAML
    pdflatex (TeX Live)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml


# ------------------------------------------------------------
# Hard-coded paths
# ------------------------------------------------------------

RULES_FOLDER = Path(
    "/home/user/Documents/Gea/cudaheat/examples/rule_laboratory/rules"
)

OUTPUT_TEX = Path(__file__).resolve().with_suffix(".tex")

TABLE_TEX = OUTPUT_TEX.with_name(
    OUTPUT_TEX.stem + "_table.tex"
)


# ------------------------------------------------------------
# LaTeX formatting
# ------------------------------------------------------------

def tex_escape(value: object) -> str:
    s = str(value or "")

    table = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    return "".join(table.get(c, c) for c in s)


def paragraph(value: object) -> str:
    text = " ".join(str(value or "").split())
    return tex_escape(text) or r"\textemdash{}"


def format_symbol(value: object) -> str:
    """Render simple rule symbols as inline mathematics."""

    symbol = str(value or "").strip()

    if not symbol:
        return r"\textemdash{}"

    if re.fullmatch(r"[A-Za-z0-9_{}^]+", symbol):
        return r"$" + symbol + r"$"

    return tex_escape(symbol)


def format_unit(value: object) -> str:
    """Convert YAML units into compact LaTeX notation."""

    unit = str(value or "").strip()

    dimensionless = {
        "",
        "1",
        "-",
        "dimensionless",
        "unitless",
        "none",
    }

    if unit.lower() in dimensionless:
        return r"-"

    replacements = {
        "degC": r"$^\circ$C",
        "degree_Celsius": r"$^\circ$C",
        "degrees_Celsius": r"$^\circ$C",
    }

    for old, new in replacements.items():
        unit = unit.replace(old, new)

    if r"$^\circ$C" in unit:
        parts = unit.split(r"$^\circ$C")
        return r"$^\circ$C".join(
            tex_escape(p) for p in parts
        )

    return tex_escape(unit)


def format_category(value: object) -> str:
    """Format primary and secondary categories."""

    if not isinstance(value, dict):
        return r"\textemdash{}"

    labels = []

    for key in ("primary", "secondary"):
        label = str(value.get(key) or "").strip()

        if label:
            labels.append(
                tex_escape(
                    label.replace("_", " ").replace("-", " ")
                )
            )

    return ", ".join(labels) or r"\textemdash{}"


def formula(value: object) -> str:
    """Preserve genuine LaTeX mathematical expressions."""

    raw = str(value or "").strip()

    if not raw:
        return r"\textemdash{}"

    raw = re.sub(r"^\s*\\\[|\\\]\s*$", "", raw).strip()
    raw = re.sub(r"^\s*\$\$|\$\$\s*$", "", raw).strip()

    chunks = [
        x.strip()
        for x in re.split(r"\n\s*\n+", raw)
        if x.strip()
    ]

    if not chunks:
        return r"\textemdash{}"

    return "\n".join(
        r"\resizebox{0.98\linewidth}{!}{$\displaystyle "
        + chunk.replace("\n", " ")
        + r"$}"
        for chunk in chunks
    )


# ------------------------------------------------------------
# Read YAML rules
# ------------------------------------------------------------

def load_rules(folder: Path) -> list[dict]:

    paths = sorted(
        (*folder.glob("*.yaml"), *folder.glob("*.yml"))
    )

    if not paths:
        raise ValueError(
            f"No YAML files found in {folder}"
        )

    entries = []

    for path in paths:

        data = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )

        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected YAML mapping"
            )

        entries.append(data)

    return entries


# ------------------------------------------------------------
# Generate table rows
# ------------------------------------------------------------

def generate_rows(entries: list[dict]) -> list[str]:

    rows = []

    for rule in entries:

        short = paragraph(
            rule.get("short_name")
        )

        long = paragraph(
            rule.get("long_name")
        )

        symbol = format_symbol(
            rule.get("symbol")
        )

        equation = formula(
            rule.get("formula_latex")
        )

        description = paragraph(
            rule.get("description")
        )

        interpretation = paragraph(
            rule.get("interpretation")
        )

        unit = format_unit(
            (rule.get("output") or {}).get("unit", "")
        )

        category = format_category(
            rule.get("category")
        )

        long_name = (
            r"{\footnotesize\textbf{"
            + symbol
            + r"} (["
            + unit
            + r"])}"
            + r"\par\vspace{2pt}"
            + r"\textbf{"
            + short
            + r"}\,--\,"
            + long
            + r"\par\vspace{2pt}"
            + r"{\scriptsize\color{metadata}\textbf{Cat}: "
            + category
            + r"}"
        )

        row = " &\n".join(
            (
                long_name,
                equation,
                description,
                interpretation,
            )
        ) + r" \\"

        rows.append(row)

    return rows


# ------------------------------------------------------------
# LaTeX main document
# ------------------------------------------------------------

def generate_main_tex(table_filename: str) -> str:

    return r"""
\documentclass[9pt,a3paper,landscape]{extarticle}

\usepackage[margin=14mm]{geometry}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}

\usepackage{
    amsmath,
    amssymb,
    graphicx,
    tabularray,
    xcolor
}

\usepackage{microtype}

% ----------------------------------------------------------
% Colors
% ----------------------------------------------------------

\definecolor{header}{HTML}{24384F}
\definecolor{ruleline}{HTML}{CAD4DE}
\definecolor{metadata}{HTML}{5B6B7C}
\definecolor{rowshade}{HTML}{F3F6F8}

% ----------------------------------------------------------
% Document
% ----------------------------------------------------------

\pagestyle{plain}

\begin{document}

\small

\input{""" + table_filename + r"""}

\end{document}
"""


# ------------------------------------------------------------
# LaTeX table
# ------------------------------------------------------------

def generate_table_tex(rows: list[str]) -> str:

    table_start = r"""
% ==========================================================
% CUDAHeat Enrichment Rules Catalogue
%
% Automatically generated from YAML definitions.
%
% Required packages:
%   tabularray
%   xcolor
%   amsmath
%   amssymb
%
% Required colors:
%   header
%   ruleline
%   metadata
%   rowshade
%
% Column widths:
%   Long name       20%
%   Formula         32%
%   Description     24%
%   Interpretation  24%
%
% ==========================================================

\begin{longtblr}[
    caption = {CUDAHeat --- Enrichment Rules Catalogue},
    entry = {CUDAHeat enrichment rules catalogue},
    label = {tab:cudaheat-rules},
]{
    colspec = {
        |Q[l,h,wd=0.20\textwidth]
        |Q[c,m,wd=0.32\textwidth]
        |Q[l,h,wd=0.24\textwidth]
        |Q[l,h,wd=0.24\textwidth]|
    },
    rowhead = 1,
    hlines = {ruleline},
    vlines = {ruleline},
    colsep = 5pt,
    rowsep = 2pt,
    row{even} = {bg=rowshade},
    row{1} = {
        bg=header,
        fg=white,
        font=\bfseries,
        valign=m,
    },
    cell{1}{2} = {halign=c},
    cell{2-Z}{2} = {halign=c,font=\scriptsize},
}
Long name & Formula & Description & Interpretation \\
"""

    table_end = r"""
\end{longtblr}
"""

    return (
        table_start
        + "\n".join(rows)
        + "\n"
        + table_end
    )


# ------------------------------------------------------------
# Generate LaTeX files
# ------------------------------------------------------------

def generate(
    folder: Path,
    target: Path,
    table_target: Path,
) -> int:

    entries = load_rules(folder)

    rows = generate_rows(entries)

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Main LaTeX document
    # --------------------------------------------------------

    main_tex = generate_main_tex(
        table_target.name
    )

    target.write_text(
        main_tex,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Separate table file
    # --------------------------------------------------------

    table_tex = generate_table_tex(rows)

    table_target.write_text(
        table_tex,
        encoding="utf-8",
    )

    return len(entries)


# ------------------------------------------------------------
# Compile LaTeX
# ------------------------------------------------------------

def compile_latex(tex_file: Path) -> None:

    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(tex_file.parent),
        str(tex_file),
    ]

    # Two passes for correct multipage table layout.
    for run in range(2):

        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=tex_file.parent,
        )

        if completed.returncode != 0:

            print(
                completed.stdout[-3500:]
            )

            raise RuntimeError(
                f"LaTeX compilation failed "
                f"on pass {run + 1}. "
                f"Inspect the .log file."
            )

    # Keep only the PDF and LaTeX sources.
    for suffix in (".aux", ".log"):
        tex_file.with_suffix(suffix).unlink(
            missing_ok=True
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    if not RULES_FOLDER.is_dir():
        raise SystemExit(
            f"Rules directory not found: {RULES_FOLDER}"
        )

    count = generate(
        RULES_FOLDER,
        OUTPUT_TEX,
        TABLE_TEX,
    )

    print(f"Generated: {OUTPUT_TEX}")
    print(f"Generated: {TABLE_TEX}")
    print(f"Rules: {count}")

    compile_latex(OUTPUT_TEX)

    print(
        f"Generated: {OUTPUT_TEX.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()
