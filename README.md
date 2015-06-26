pyalacarte 
==========
**_A python 3 implementation of the A la Carte large scale GP_**

- **Authors**: Daniel Steinberg, Alistair Reid, Lachlan McCalman
- **Institute**: NICTA
- **Date**: 26 June 2015

Have a look at `demos/alacarte_demo.py` for how this compares to a normal GP.
Have a go at tweaking the parameters of this script too.

To run the demo, just call `./alaracate_demo.py` *if you have installed this
package*. Otherwise, from the root directory of this package, run
`./demos/alacarte_demo.py`.

## Requirements:
- Numpy
- Scipy
- NLopt
- bdkd-external (https://github.com/NICTA/bdkd-external) for the demo


## References:

[1] Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte -- Learning 
    Fast Kernels". Proceedings of the Eighteenth International Conference on
    Artificial Intelligence and Statistics, pp. 1098–1106, 2015.
