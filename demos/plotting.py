import matplotlib.pyplot as pl


def fancy_yyplot(y, Ey, lowers, uppers, varname):

    for k in range(len(y)):
        pl.plot([y[k], y[k]],
                [lowers[k], uppers[k]],
                'b', alpha=0.25, linewidth=5)

    pl.plot(y, Ey, 'o', color='#475F83', linewidth=2, markersize=5)

    maxy = max(y.max(), Ey.max())
    miny = min(y.min(), Ey.min())
    yrange = [miny, maxy]
    pl.plot([miny, maxy], [miny, maxy], color='#C76C63', linewidth=3)
    pl.xlim(yrange)
    pl.ylim(yrange)
    title = 'Y-Y plot for {}'.format(varname)
    pl.grid(True)
    pl.title(title)
    pl.xlabel('True')
    pl.ylabel('Predicted')
    pl.show()


def yyplot(y, Ey, varname):

    pl.plot(y, Ey, '.', color='#475F83', linewidth=2)
    maxy = max(y.max(), Ey.max())
    miny = min(y.min(), Ey.min())
    pl.plot([miny, maxy], [miny, maxy], color='#C76C63', linewidth=3)

    title = 'Y-Y plot for {}'.format(varname)

    pl.grid(True)
    pl.title(title)
    pl.xlabel('True')
    pl.ylabel('Predicted')
    pl.show()


def SGDtrace(objs, grad_norms, fig):

    maxiter = len(objs)
    ax = fig.add_subplot(111)
    pl.xlabel('Iteration')

    pl.title('SGD convergence')
    ax.plot(range(maxiter), grad_norms, 'r')
    ax.set_ylabel('gradient norms', color='r')

    for t in ax.get_yticklabels():
        t.set_color('r')

    ax2 = ax.twinx()
    ax2.plot(range(maxiter), objs, 'b')
    ax2.set_ylabel('Objective', color='b')

    for t in ax2.get_yticklabels():
        t.set_color('b')

    pl.show()
