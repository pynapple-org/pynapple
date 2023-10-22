#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_namespace_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

# with open('docs/HISTORY.md') as history_file:
#     history = history_file.read()

requirements = [
    'pandas>=1.0.3',
    'numba>=0.46.0',
    'numpy>=1.17.4',
    'scipy>=1.3.2',
    'pynwb>=2.0.0',
    'tabulate',
    'h5py',
    'tifffile',
    'zarr',
    'rich'
]

test_requirements = [
    'pytest',
    'isort',
    'pip-tools',
    'flake8',
    'coverage'
]

setup(
    author="Guillaume Viejo",
    author_email='guillaume.viejo@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="PYthon Neural Analysis Package Pour Laboratoires d’Excellence",
    install_requires=requirements,
    license="MIT License",
    # long_description='pynapple is a Python library for analysing neurophysiological data. It allows to handle time series and epochs but also to use generic functions for neuroscience such as tuning curves and cross-correlogram of spikes. It is heavily based on neuroseries.' 
    # + '\n\n' + history,
    long_description=readme,
    include_package_data=True,
    keywords='neuroscience',
    name='pynapple',    
    packages=find_namespace_packages(
        include=['pynapple', 'pynapple.*'],
        exclude=['tests']
        ),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/pynapple-org/pynapple',
    version='v0.4.0',
    zip_safe=False,
    long_description_content_type='text/markdown',
    download_url='https://github.com/pynapple-org/pynapple/archive/refs/tags/v0.4.0.tar.gz'
)
