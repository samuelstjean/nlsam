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
                                  'nibabel>=2.2',
                                  'spams>=2.4']
    params['dependency_links'] = ['https://github.com/samuelstjean/spams-python/releases/download/0.1/spams-2.6.zip#egg=spams-2.6']

    return params
