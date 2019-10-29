#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${DIR}" || exit

jupyter nbconvert --to latex Neuron_types_example.ipynb
sed -r -i 's/documentclass\[11pt\]\{article\}/documentclass[8pt]{extarticle}/' Neuron_types_example.tex
sed -r -i 's/geometry\{verbose,tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in}/geometry{verbose,tmargin=0.5in,bmargin=0.5in,lmargin=0.2in,rmargin=0.2in}/' Neuron_types_example.tex
pdflatex Neuron_types_example.tex