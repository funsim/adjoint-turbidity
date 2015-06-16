#!/usr/bin/python
from pickle import *
from numpy import *
import glob

y_offset = 0.1

file = "final.pckl"
data = load(open(file))

min_x = 1000000
max_x = -1000000
for x in 'data_x', 'fn_x':
    min_x = min(min_x, data[x].min())
    max_x = max(max_x, data[x].max())
min_y = 1000000
max_y = -1000000
for x in 'data_y', 'result', 'target':
    min_y = min(min_y, data[x].min())
    max_y = max(max_y, data[x].max())

# header
print '''\\documentclass{standalone}
\\usepackage{pgfplots}
\\usepackage{amsmath}
\\usetikzlibrary{calc}
\\pgfplotsset{compat=newest}
\\pgfplotsset{every axis/.append style={line width=0.5pt}}
\\pgfplotsset{every axis legend/.append style={
font=\\footnotesize,
at={(0.99,0.99)},
anchor=north east}}
\\tikzset{mark size=1.0}
\\begin{document}
\\begin{tikzpicture}
\\begin{axis}[
width=0.75\\textwidth,
height=1.0\\textwidth,
font=\scriptsize,
scaled ticks=false,
xmin = %.15f,
xmax = %.15f,
ymin = %.15f,
ymax = %.15f,
ylabel=deposit depth,
xlabel=$x$ $(\\times 10^{3})$,
line width=0.1pt
]'''%(0, max_x/1000, 0, max_y + y_offset)

# xticklabel style={
# anchor=north,
# /pgf/number format/precision=1,
# /pgf/number format/sci,
# /pgf/number format/sci zerofill,
# /pgf/number format/sci generic={mantissa sep=\\times,exponent={10^{#1}}}
# }

print '''\\addplot[densely dashed, only marks, mark=x, mark options=solid] table {
x y'''
for point in zip(data['data_x'], data['data_y']):
    print point[0]/1000, point[1]
print '''};
\\addlegendentry{field data}'''

print '''\\addplot[densely dashed] table {
x y'''
for point in zip(data['fn_x'], data['target']):
    print point[0]/1000, point[1]
print '''};
\\addlegendentry{$\\eta_T$}'''

print '''\\addplot[solid] table {
x y'''
for point in zip(data['fn_x'], data['result']):
    print point[0]/1000, point[1]
print '''};
\\addlegendentry{$\\eta$}'''

# print '''\\addplot[solid, red] table {
# x y'''
# for point in zip(data['fn_x'], data['functional']):
#     print point[0], point[1]**0.5
# print '''};
# \\addlegendentry{$| \\eta - \\eta_T |$}'''

print '''\\end{axis}
\\end{tikzpicture}
\\end{document}'''
