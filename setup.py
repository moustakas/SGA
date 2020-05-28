#!/usr/bin/env python

# Supports:
# - python setup.py install
#
# Does not support:
# - python setup.py test
# - python setup.py version

import os, glob
from setuptools import setup, find_packages

def _get_version():
    import subprocess
    version = subprocess.check_output('git describe', shell=True)
    version = version.decode('utf-8').replace('\n', '')
    return version

version = _get_version()

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup_kwargs=dict(
    name='SGA',
    url='https://github.com/moustakas/SGA',
    version=version,
    author='John Moustakas',
    author_email='jmoustakas@siena.edu',
    #packages=[],
    license=license,
    description='Siena Galaxy Atlas',
    long_description=readme,
)

#- What to install
setup_kwargs['packages'] = find_packages('py')
setup_kwargs['package_dir'] = {'':'py'}

#- Treat everything in bin/ as a script to be installed
setup_kwargs['scripts'] = glob.glob(os.path.join('bin', '*'))

#- Data to include
# setup_kwargs['package_data'] = {
#     'legacyhalos': ['data/*',],
#     'legacyhalos.test': ['data/*',],
# }

#- Testing
#setup_kwargs['test_suite'] = 'SGA.test.test_suite'

#- Go!
setup(**setup_kwargs)
