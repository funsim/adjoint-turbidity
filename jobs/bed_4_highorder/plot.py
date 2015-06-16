#!/usr/bin/python
from pickle import *
from numpy import *
import glob

s = 1.580126e-01
method = "IPOPT"
offset = 0.2
t_offset = 0.01

n = [eval(f_.split('_')[-1].split('.')[0]) for f_ in glob.glob('opt_%s*.pckl'%method)]
n.sort()
n = n[:15]
files = ['opt_%s_%d.pckl'%(method, i) for i in n]
#print files[-1]

min_ = 1000000
max_ = -1000000
for f in files:
    a = load(open(f))
    min_ = min(min_, (array(a['ic'])*array([[1],[1],[1]])).min())
    max_ = max(max_, (array(a['ic'])*array([[1],[1],[1]])).max())

# header
print '''\\documentclass{standalone}
\\usepackage{pgfplots}
\\usepackage{amsmath}
\\usetikzlibrary{calc}
\\pgfplotsset{compat=newest}
\\pgfplotsset{every axis/.append style={line width=0.5pt}}
\\pgfplotsset{every axis legend/.append style={
font=\\footnotesize,
at={(0.01,0.99)},
anchor=north west}}
\\tikzset{mark size=1.0}
\\begin{document}
\\begin{tikzpicture}
\\begin{semilogyaxis}[
width=0.75\\textwidth,
height=1.0\\textwidth,
font=\scriptsize,
scaled ticks=false,
xmin = %.15f,
xmax = %.15f,
yticklabel style={
anchor=east,
/pgf/number format/precision=2,
/pgf/number format/fixed,
/pgf/number format/fixed zerofill,
},
ylabel=$J$,
line width=0.1pt,
clip=false
]'''%(-offset, max_ + offset)

print '''\\addplot[red, densely dashed, mark=*, mark options=solid] table[x=var,y=j] {
var j'''
for f in files:
    a = load(open(f))
    print a['ic'][0][0], a['j']*s
print '''};
\\addlegendentry{$(h_0)_n / (h_0)_0$}'''

print '''\\addplot[blue, densely dashed, mark=*, mark options=solid] table[x=var,y=j] {
var j'''
for f in files:
    a = load(open(f))
    print a['ic'][1][0], a['j']*s
print '''};
\\addlegendentry{$(\\psi_0)_n / (\\psi_0)_0$}'''

print '''\\addplot[black!30!green, densely dashed, mark=*, mark options=solid] table[x=var,y=j] {
var j'''
for f in files:
    a = load(open(f))
    print a['ic'][2][0]**0.5, a['j']*s
print '''};
\\addlegendentry{$(D)_n / (D)_0$}'''

a = load(open('opt_%s_1.pckl'%method))
print "\\node[anchor=west] at (axis cs:%.15f,%.15f) {\\tiny iteration $(n)$};"%(max_ + offset + t_offset, 5e-2)

j = 0
for i in n:
  a = load(open('opt_%s_%d.pckl'%(method, i)))
  print "\\addplot[gray, densely dotted, domain=%.15f:%.15f] function { %.15f };"%(- offset, max_ + offset, a['j']*s)
  print "\\node[anchor=west] at (axis cs:%.15f,%.15f) {\\tiny $%d$};"%(max_ + offset + t_offset, a['j']*s, j)
  j+=1


print '''\\end{semilogyaxis}
\\end{tikzpicture}
\\end{document}'''
