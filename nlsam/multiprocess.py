import os

try:
    from multiprocessing import get_context
    has_context = True
except ImportError:
    from multiprocessing import Pool
    has_context = False

try:
    import mkl
    has_mkl = True
except ImportError:
    has_mkl = False


def multiprocessing_hanging_workaround():
    # Fix openblas threading bug with openmp before loading numpy
    # Spams has openmp support already, and openblas conflicts with python multiprocessing.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # Mac OSX has it's own blas/lapack, but like openblas it causes conflict for
    # python 2.7 and before python 3.4 in multiprocessing, so disable it.
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'


def multiprocesser(func, args, n_cores=None, mp_method=None):
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

    Returns
    -------
    output : list
        Each element is the result of func(*args)

    Notes
    ----
    See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    for more information on available starting methods.
    It also seems to cause issues if called from a cython function on the buildbots, but not on my computer,
    so use with caution (or just use the old fashioned pool method directly from cython).
    '''

    # we set mkl to only use one core in multiprocessing, then restore it back afterwards
    if has_mkl:
        # This is only 4 on my quad core laptop, with hyper threading though :/
        # I guess the intel guys know what is best anyway
        mkl_threads = mkl.get_max_threads()
        mkl.set_num_threads(1)

    # Only python >= 3.4, so if we don't have it go back to the old fashioned Pool
    if has_context:
        with get_context(method=mp_method).Pool(processes=n_cores) as pool:
            output = pool.map(func, args)
    else:
        pool = Pool(processes=n_cores)
        output = pool.map(func, args)
        pool.close()
        pool.join()

    if has_mkl:
        mkl.set_num_threads(mkl_threads)

    return output
