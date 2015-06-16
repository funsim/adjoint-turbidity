#!/usr/bin/python
from pickle import *
from numpy import *
import glob
from target import fit

y_offset = 0.01
x_start = 3.34820228667

files = "final.pckl", "final_end.pckl"
data = [load(open(file)) for file in files]

min_x = 1000000
max_x = -1000000
for x in 'data_x', 'fn_x':
  for d in data:
    min_x = min(min_x, d[x].min())
    max_x = max(max_x, d[x].max())
min_y = 1000000
max_y = -1000000
for x in 'data_y', 'result', 'target':
  for d in data:
    min_y = min(min_y, d[x].min())
    max_y = max(max_y, d[x].max())

# header
print '''\\documentclass{standalone}
\\usepackage{pgfplots}
\\usepackage{amsmath}
\\usetikzlibrary{calc}
\\pgfplotsset{compat=newest}
\\pgfplotsset{every axis/.append style={line width=0.5pt}}
\\pgfplotsset{every axis legend/.append style={
font=\\small,
at={(0.99,0.99)},
anchor=north east}}
\\tikzset{mark size=1.0}
\\begin{document}
\\begin{tikzpicture}
\\begin{axis}[
width=0.75\\textwidth,
height=0.75\\textwidth,
font=\small,
scaled ticks=false,
xmin = %.15f,
xmax = %.15f,
ymin = %.15f,
ymax = %.15f,
ylabel=deposit depth,
xlabel=$x$ $(\\times 10^{3})$,
line width=0.5pt,
yticklabel style={
anchor=east,
/pgf/number format/precision=2,
/pgf/number format/fixed,
/pgf/number format/fixed zerofill
}
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
for point in zip(data[0]['data_x'], data[0]['data_y']):
    print point[0]/1000, point[1]
print '''};
\\addlegendentry{field data}'''

print '''\\addplot[densely dashed, red] table {
x y'''
ec_coeff = fit(5)
for x in linspace(0, data[0]['data_x'][-1], 100):
    y = 0
    for i, c in enumerate(ec_coeff):
        y += c((0,0))*pow(x, i)
    print x/1000, y
print '''};
\\addlegendentry{$\\eta_T$ (target profile)}'''

print '''\\addplot[dashdotted] table {
x y'''
x_mod = data[0]['fn_x'] - 3.34820228667**1.0*2300
for point in zip(x_mod, data[0]['result']):
    print point[0]/1000, point[1]
print '''};
\\addlegendentry{$\\tilde{\\eta}_0$ (non-optimised profile)}'''

print '''\\addplot[solid] table {
x y'''
x_mod = data[1]['fn_x'] - 3.34820228667**1.1145998528775172*2300
for point in zip(x_mod, data[1]['result']):
    print point[0]/1000, point[1]
print '''};
\\addlegendentry{$\\tilde{\\eta}$ (optimised profile)}'''

# print '''\\addplot[solid, red] table {
# x y'''
# for point in zip(data['fn_x'], data['functional']):
#     print point[0], point[1]**0.5
# print '''};
# \\addlegendentry{$| \\eta - \\eta_T |$}'''

print '''\\end{axis}
\\end{tikzpicture}
\\end{document}'''
