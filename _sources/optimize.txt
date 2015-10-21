Optimization
============

==============  =================================
scipy.optimize  NLopt
==============  =================================
Nelder-Mead     LN_NELDERMEAD
Powell          ?
CG              ?
BFGS            LD_LBFGS
Newton-CG       | **LD_TNEWTON_PRECOND_RESTART**
                | LD_TNEWTON_PRECOND
                | LD_TNEWTON_RESTART
                | LD_TNEWTON
L-BFGS-B        LD_LBFGS
TNC             | **LD_TNEWTON_PRECOND_RESTART**
                | LD_TNEWTON_PRECOND
                | LD_TNEWTON_RESTART
                | LD_TNEWTON
COBYLA          LN_COBYLA
SLSQP           LD_SLSQP
dogleg          *Requires second-order gradients*
trust-ncg       *Requires second-order gradients*
==============  =================================

.. automodule:: pyalacarte.optimize
   :members:
