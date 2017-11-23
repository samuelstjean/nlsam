import os
from itertools import repeat

try:
    from itertools import izip
except ImportError:  # Python 3 built-in zip already returns iterable
    izip = zip

try:
    from multiprocessing import get_context
    has_context = True
except ImportError:
    from multiprocessing import Pool
    has_context = False


def multiprocessing_hanging_workaround():
    # Fix openblas threading bug with openmp before loading numpy
    # Spams has openmp support already, and openblas conflicts with python multiprocessing.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # Mac OSX has it's own blas/lapack, but like openblas it causes conflict for
    # python 2.7 and before python 3.4 in multiprocessing, so disable it.
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'


def _func_star(stuff):
    """First argument is the function to run in parallel, second is the arglist of stuff
    """
    return stuff[0](*list(stuff[1]))


def multiprocesser(func, n_cores=None, mp_method=None):
    '''
    Runs func in parallel by unpacking each element of arglist.

    Parameters
    ----------
    func : function
        A function which accepts a single tuple (which will be unpacked) as an argument.
    args : list
        A list of each tuples to unpack into func.
    n_cores : int
        Number of processes to launch at the same time.
    mp_method : string
        Dispatch method for multiprocessing, see Notes for more information.
        Has an effect only for python 3.4 and later.

    Returns
    -------
    output : list
        Each element is the result of func(*args)

    Notes
    ----
    See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    for more information on available starting methods.
    It also seems to cause issues on the buildbots, but not on my computer,
    so use with caution (it seems that maybe virtualenv/the way travis launches jobs
    conflicts with it, not sure what is the actual problem).
    '''

    # Only python >= 3.4, so if we don't have it go back to the old fashioned Pool
    if has_context:
        def parfunc(args):
            with get_context(method=mp_method).Pool(processes=n_cores) as pool:
                output = pool.map(_func_star, izip(repeat(func), args))
            return output
    else:
        def parfunc(args):
            pool = Pool(processes=n_cores)
            output = pool.map(_func_star, izip(repeat(func), args))
            pool.close()
            pool.join()
            return output

    return parfunc
