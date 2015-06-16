#!/usr/bin/python
from pickle import *
from numpy import *
import glob
from target import fit

y_offset = 0.1

file0 = "final.pckl"
file1 = "../bed_4/final.pckl"
data0 = load(open(file0))
data1 = load(open(file1))

min_x = 1000000
max_x = -1000000
for x in 'data_x', 'fn_x':
    min_x = min(min_x, data0[x].min())
    max_x = max(max_x, data0[x].max())
    min_x = min(min_x, data1[x].min())
    max_x = max(max_x, data1[x].max())
min_y = 1000000
max_y = -1000000
for x in 'data_y', 'result', 'target':
    min_y = min(min_y, data0[x].min())
    max_y = max(max_y, data0[x].max())
    min_y = min(min_y, data1[x].min())
    max_y = max(max_y, data1[x].max())

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
width=1.0\\textwidth,
height=0.75\\textwidth,
font=\scriptsize,
scaled ticks=false,
xmin = %.15f,
xmax = %.15f,
ymin = %.15f,
ymax = %.15f,
ylabel=deposit depth (m),
xlabel=$x$ $(\\times 10^{3}\mathrm{m})$,
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
for point in zip(data0['data_x'], data0['data_y']):
    print point[0]/1000, point[1]
print '''};
\\addlegendentry{field data}'''

print '''\\addplot[densely dashed, red] table {
x y'''
from target import fit
ec_coeff = fit(2, 0)
for x in linspace(0, data0['data_x'][-1], 100):
    y = 0
    for i, c in enumerate(ec_coeff):
        y += c((0,0))*pow(x, i)
    print x/1000, y
print '''};
\\addlegendentry{$\\eta_T^1$ (target profile 1)}'''

print '''\\addplot[densely dashed, blue] table {
x y'''
ec_coeff = fit(11, 0)
for x in linspace(0, data0['data_x'][-1], 100):
    y = 0
    for i, c in enumerate(ec_coeff):
        y += c((0,0))*pow(x, i)
    print x/1000, y
print '''};
\\addlegendentry{$\\eta^T_{10}$ (target profile 2)}'''

print '''\\addplot[solid, red] table {
x y'''
for point in zip(data1['fn_x'], data1['result']):
    print point[0]/1000, point[1]
print '''};
\\addlegendentry{$\\eta_1$ (optimised profile 1)}'''

print '''\\addplot[solid, blue] table {
x y'''
for point in zip(data0['fn_x'], data1['result']):
    print point[0]/1000, point[1]
print '''};
\\addlegendentry{$\\eta_{10}$ (optimised profile 2)}'''

# print '''\\addplot[solid, red] table {
# x y'''
# for point in zip(data['fn_x'], data['functional']):
#     print point[0], point[1]**0.5
# print '''};
# \\addlegendentry{$| \\eta - \\eta_T |$}'''

print '''\\end{axis}
\\end{tikzpicture}
\\end{document}'''
