#!/usr/bin/env python3
"""Generate a LaTeX catalogue of CUDAHeat YAML rules.

Usage:
    python pythonScripts/generate_rules_catalogue.py
    python pythonScripts/generate_rules_catalogue.py /path/to/rules /path/to/LaTex/tables
    python pythonScripts/generate_rules_catalogue.py /path/to/rules /path/to/LaTex/tables --compile
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml

LATEX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LATEX_ROOT.parent
DEFAULT_RULES_FOLDER = PROJECT_ROOT / "rules"
DEFAULT_OUTPUT_DIRECTORY = LATEX_ROOT / "tables"

printFlag=False


def tex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value or ""))


def paragraph(value: object) -> str:
    return tex_escape(" ".join(str(value or "").split())) or r"\textemdash{}"


def format_symbol(value: object) -> str:
    symbol = str(value or "").strip()
    if not symbol:
        return r"\textemdash{}"
    if re.fullmatch(r"[A-Za-z0-9_{}^]+", symbol):
        return "$" + symbol + "$"
    return tex_escape(symbol)


def format_unit(value: object) -> str:
    unit = str(value or "").strip()
    if unit.lower() in {"", "1", "-", "dimensionless", "unitless", "none"}:
        return "-"
    for name in ("degC", "degree_Celsius", "degrees_Celsius"):
        unit = unit.replace(name, "__DEGC__")
    return tex_escape(unit).replace(r"\_\_DEGC\_\_", r"$^\circ$C")


def format_category(value: object) -> str:
    if not isinstance(value, dict):
        return r"\textemdash{}"
    labels = [
        tex_escape(str(value.get(key) or "").strip().replace("_", " ").replace("-", " "))
        for key in ("primary", "secondary")
        if str(value.get(key) or "").strip()
    ]
    return ", ".join(labels) or r"\textemdash{}"


def formula(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return r"\textemdash{}"
    raw = re.sub(r"^\s*\\\[|\\\]\s*$", "", raw).strip()
    raw = re.sub(r"^\s*\$\$|\$\$\s*$", "", raw).strip()
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", raw) if chunk.strip()]
    return "\n".join(
        r"\resizebox{0.98\linewidth}{!}{$\displaystyle "
        + chunk.replace("\n", " ") + r"$}"
        for chunk in chunks
    ) or r"\textemdash{}"


def load_rules(folder: Path) -> list[dict]:
    paths = sorted((*folder.glob("*.yaml"), *folder.glob("*.yml")))
    if not paths:
        raise ValueError(f"No YAML files found in {folder}")
    entries = []
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected YAML mapping")
        entries.append(data)
    return entries


def generate_rows(entries: list[dict]) -> list[str]:
    rows = []
    for rule in entries:
        short = paragraph(rule.get("short_name"))
        long = paragraph(rule.get("long_name"))
        symbol = format_symbol(rule.get("symbol"))
        equation = formula(rule.get("formula_latex"))
        description = paragraph(rule.get("description"))
        interpretation = paragraph(rule.get("interpretation"))
        unit = format_unit((rule.get("output") or {}).get("unit", ""))
        category = format_category(rule.get("category"))
        long_name = (
            r"{\footnotesize\textbf{" + symbol + r"} ([" + unit + r"])}"
            + r"\par\vspace{2pt}\textbf{" + short + r"}\,--\,"
            + long + r"\par\vspace{2pt}"
            + r"{\scriptsize\color{metadata}\textbf{Cat}: " + category + r"}"
        )
        rows.append(" &\n".join((long_name, equation, description, interpretation)) + r" \\")
    return rows


def generate_table_tex(rows: list[str]) -> str:
    start = r"""\begin{longtblr}[
    caption = {CUDAHeat --- Enrichment Rules Catalogue},
    entry = {CUDAHeat enrichment rules catalogue},
    label = {tab:cudaheat-rules},
]{
    colspec = {
        |Q[l,h,wd=0.2\textwidth]
        |Q[c,m,wd=0.4\textwidth]
        |Q[l,h,wd=0.25\textwidth]
        |Q[l,h,wd=0.25\textwidth]|
    },
    rowhead = 1,
    hlines = {ruleline},
    vlines = {ruleline},
    colsep = 5pt,
    rowsep = 2pt,
    row{even} = {bg=rowshade},
    row{1} = {bg=header, fg=white, font=\bfseries, valign=m},
    cell{1}{2} = {halign=c},
    cell{2-Z}{2} = {halign=c,font=\scriptsize},
}
Long name & Formula & Description & Interpretation \\
"""
    return start + "\n".join(rows) + "\n" + r"\end{longtblr}" + "\n"


def generate_main_tex(table_filename: str) -> str:
    return r"""\documentclass[9pt,a3paper,landscape]{extarticle}
\usepackage[margin=14mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,graphicx,tabularray,xcolor}
\usepackage{microtype}
\definecolor{header}{HTML}{24384F}
\definecolor{ruleline}{HTML}{CAD4DE}
\definecolor{metadata}{HTML}{5B6B7C}
\definecolor{rowshade}{HTML}{F3F6F8}
\pagestyle{plain}
\begin{document}
\small
\input{""" + table_filename + "}\n" + r"\end{document}" + "\n"


def compile_latex(tex_file: Path) -> None:
    command = [
        "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
        "-output-directory", str(tex_file.parent), tex_file.name,
    ]
    for pass_number in (1, 2):
        completed = subprocess.run(command, cwd=tex_file.parent, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(
                f"LaTeX compilation failed on pass {pass_number}:\n{completed.stdout[-3500:]}"
            )
    for suffix in (".aux", ".log"):
        tex_file.with_suffix(suffix).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", nargs="?", type=Path, default=DEFAULT_RULES_FOLDER)
    parser.add_argument("output_directory", nargs="?", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    source = args.input_directory.expanduser().resolve()
    destination = args.output_directory.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Rules directory not found: {source}")
    entries = load_rules(source)
    destination.mkdir(parents=True, exist_ok=True)
    table_file = destination / "generate_rules_catalogue_table.tex"
    document_file = destination / "generate_rules_catalogue.tex"
    table_file.write_text(generate_table_tex(generate_rows(entries)), encoding="utf-8")
    document_file.write_text(generate_main_tex(table_file.name), encoding="utf-8")
    if printFlag: 
        print(f"Input: {source}")
        print(f"Rules: {len(entries)}")
        print(f"Table: {table_file}")
        print(f"Document: {document_file}")
    if args.compile and printFlag:
        compile_latex(document_file)
        print(f"PDF: {document_file.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
