# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='LSLGA',
    url='https://github.com/moustakas/LSLGA',
    version='untagged',
    author='John Moustakas',
    author_email='jmoustakas@siena.edu',
    #packages=[],
    license=license,
    description='Legacy Survey Large Galaxy Atlas',
    long_description=readme,
    #package_data={},
    #scripts=,
    #include_package_data=True,
    #install_requires=['numpy']
)
