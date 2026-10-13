# CUDAHeat Beamer theme

Place beamerthemeCUDAHeat.sty beside your .tex file and use:

```latex
\documentclass[aspectratio=169]{beamer}
\usetheme{CUDAHeat}
\begin{document}
\begin{CUDAHeatFrame}{Title}{Subtitle}{Takeaway}{Source}
  \CUDAHeatPanelLabel{PANEL LABEL}
  \CUDAHeatContent{Your content here.}
\end{CUDAHeatFrame}
\end{document}
```

Compile with XeLaTeX or LuaLaTeX; install DejaVu Sans and TikZ.
The example also uses PGFPlots. Compile `xelatex example.tex` twice.

The theme sets the physical canvas to the original 13.32 × 7.4925 inches.
This preserves the source font sizes as well as its 16:9 ratio. Header,
title, subtitle, white panel, takeaway, and source use the Python layout's
normalized coordinates. All eight colour definitions retain their hex values;
CUDAHeatWhite is also provided. TeX's text metrics and rounded corner geometry
may differ slightly from Matplotlib.

Use CUDAHeatFrame for the designed slides; a bare frame does not create this
custom title/subtitle/caption layout. Its body accepts ordinary LaTeX.
CUDAHeatContent places a top-aligned text box inside the panel.
CUDAHeatPlace{x}{y}{content} anchors any content's bottom-left corner at a
normalized canvas position. The source's default chart box starts at
(0.125, 0.267), width 0.735 and height 0.35 of the canvas.
Keep long captions concise: the original layout uses single lines.
CUDAHeatSetBrand{...} changes the small header globally.

The example chart is explicitly illustrative and contains no measured data.
The theme itself neither reads JSON nor exports Matplotlib graphics.
