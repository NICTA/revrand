Demos
=====

.. ipython::

   @verbatim
   In [0]: from matplotlib import pyplot as plt
      ...: import numpy as np

   In [0]: x = np.linspace(0, 5, 10)

   In [0]: y = x ** 2

   @savefig plot.png width=4in
   In [0]: fig, axes = plt.subplots()
      ...: axes.plot(x, y, 'r')
      ...: axes.set_xlabel('x')
      ...: axes.set_ylabel('y')
      ...: axes.set_title('title')
      ...: plt.show();