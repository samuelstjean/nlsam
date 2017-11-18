import sys
import os


def fix_multiproc_windows():
    # See https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
    # Module multiprocessing is organized differently in Python 3.4+
    try:
        # Python 3.4+
        if sys.platform.startswith('win'):
            import multiprocessing.popen_spawn_win32 as forking
        else:
            import multiprocessing.popen_fork as forking
    except ImportError:
        import multiprocessing.forking as forking

    if sys.platform.startswith('win'):
        # First define a modified version of Popen.
        class _Popen(forking.Popen):
            def __init__(self, *args, **kw):
                if hasattr(sys, 'frozen'):
                    # We have to set original _MEIPASS2 value from sys._MEIPASS
                    # to get --onefile mode working.
                    os.putenv('_MEIPASS2', sys._MEIPASS)
                try:
                    super(_Popen, self).__init__(*args, **kw)
                finally:
                    if hasattr(sys, 'frozen'):
                        # On some platforms (e.g. AIX) 'os.unsetenv()' is not
                        # available. In those cases we cannot delete the variable
                        # but only set it to the empty string. The bootloader
                        # can handle this case.
                        if hasattr(os, 'unsetenv'):
                            os.unsetenv('_MEIPASS2')
                        else:
                            os.putenv('_MEIPASS2', '')

        # Second override 'Popen' class with our modified version.
        forking.Popen = _Popen


def get_setup_params():
    params = {}
    params['scripts'] = ['scripts/nlsam_denoising']
    params['name'] = 'nlsam'
    params['author'] = 'Samuel St-Jean'
    params['author_email'] = 'samuel@isi.uu.nl'
    params['url'] = 'https://github.com/samuelstjean/nlsam'
    params['version'] = '0.6.1'
    params['install_requires'] = ['numpy>=1.10.4',
                                  'scipy>=0.19.1',
                                  'cython>=0.21',
                                  'nibabel>=2.0',
                                  'spams>=2.4']
    params['dependency_links'] = ['https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-2.6.zip#egg=spams-2.6']

    return params


if sys.platform.startswith('win'):
    fix_multiproc_windows()
