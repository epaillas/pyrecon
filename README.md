# pyrecon - Python reconstruction code

## Introduction

**pyrecon** is a package to perform reconstruction within Python, using different algorithms, so far:

 - MultiGridReconstruction, based on Martin White's code https://github.com/martinjameswhite/recon_code
 - IterativeFFTReconstruction, based on Julian Bautista's code https://github.com/julianbautista/eboss_clustering/blob/master/python/recon.py


With Python, a typical reconstruction run is (e.g. for MultiGridReconstruction; the same works for other algorithms):
```
from pyrecon import MultiGridReconstruction

rec = MultiGridReconstruction(f=0.8,bias=2.0,nmesh=512,boxsize=1000.,boxcenter=2000.)
rec.assign_data(positions_data,weights_data)
rec.assign_randoms(positions_randoms,weights_randoms)
rec.set_density_contrast()
rec.run()
positions_rec_data = positions_data - rec.read_shifts(positions_data)
# RecSym = remove large scale RSD from randoms
positions_rec_randoms = positions_randoms - rec.read_shifts(positions_randoms)
# Or RecIso
# positions_rec_randoms = positions_randoms - rec.read_shifts(positions_randoms,with_rsd=False)
```
Also provided a script to run reconstruction as a standalone:
```
pyrecon [-h] config-fn [--data-fn [<fits, hdf5 file>]] [--randoms-fn [<fits, hdf5 file>]] [--output-data-fn [<fits, hdf5 file>]] [--output-randoms-fn [<fits, hdf5file>]]
```
An example of configuration file is provided in [config]https://github.com/adematti/pyrecon/blob/main/bin/config_example.yaml.
data-fn, randoms-fn are input data and random file names to override those in configuration file.
The same holds for output files output-data-fn, output-randoms-fn.

## Notes

Numerical agreement in the Zeldovich displacements between original codes and this new reimplementation is:

  - IterativeFFTReconstruction: absolute and relative difference of 1e-12 (machine precision)
  - MultiGridReconstruction: 2e-4 (absolute), 2e-3 (relative) difference. Switching from float32 to float64 in the new implementation produces shifts of 1e-3, so it is probably simply a
    matter of numerical accuracy.

## In progress

Check algorithm details (see notes in docstrings).

## Documentation

Documentation is hosted on Read the Docs, [pyrecon docs](https://pyrecon.readthedocs.io/).

# Requirements

Only strict requirements are:
- numpy

For faster FFTs:
- pyfftw

## Installation

To install the code:
```
$>  python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately):
```
$>  python setup.py develop --user
```

With Mac OS, if you wish to use clang compiler (instead of gcc), you may encounter an error related to ``-fopenmp`` flag.
In this case, you can try to export:
```
$>  export CC=clang
```
Before installing pyrecon. This will set clang OpenMP flags for compilation.
Note that with Mac OS gcc can point to clang.

## License

**pyrecon** is free software distributed under a GPLv3 license. For details see the [LICENSE](https://github.com/adematti/pyrecon/blob/main/LICENSE).

## Credits

Martin J. White and Julian E. Bautista for their codes, Pedro Rangel Caetano for inspiration for the script bin/recon.