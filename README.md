# Regulation of T cell expansion by antigen presentation dynamics

This repository contains the source code associated with the manuscript

Mayer, Zhang, Perelson, Wingreen: [Regulation of T cell expansion by antigen presentation dynamics](https://doi.org/10.1073/pnas.1812800116), PNAS 2019

It allows reproduction of all numerical results reported in the manuscript.

![Figure 1](https://raw.githubusercontent.com/andim/paper-tcellexp/master/fig1_S234/fig1final.png "Limitation of T cell expansion by antigen decay can explain the power-law dependence of fold expansion on the initial number of cognate T cells")

## Installation requirements

The code uses Python 3.6+.

A number of standard scientific python packages are needed for the numerical simulations and visualizations. An easy way to install all of these is to install a Python distribution such as [Anaconda](https://www.continuum.io/downloads). 

- [numpy](http://github.com/numpy/numpy/)
- [scipy](https://github.com/scipy/scipy)
- [pandas](http://github.com/pydata/pandas)
- [matplotlib](http://github.com/matplotlib/matplotlib)

Additionally the code also relies on this package:

- [projgrad](https://github.com/andim/projgrad)

## Structure/running the code

Every folder contains a file `plot.py` which needs to be run to produce the figures. For a number of figures cosmetic changes were done in inkscape as a postprocessing step. In these cases the figures will not be reproduced precisely. To help reuse the final edited figures are provided in png/svg format.

## Contact

If you run into any difficulties running the code, feel free to contact us at `andimscience@gmail.com`.

## License

The source code is freely available under an MIT license. The plots are licensed under a Creative commons attributions license (CC-BY).
