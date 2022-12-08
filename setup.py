#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('docs/HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
        'pandas>=1.0.3',
        'numba>=0.46.0',
        'numpy>=1.17.4',
        'scipy>=1.3.2',
        'pynwb',
        'tabulate',
        'pyqt5',
        'pyqtgraph',
        'h5py',
        'tifffile',
        'zarr'
        ]

test_requirements = [
    'pytest',
    'isort',
    'pip-tools',
    'pytest',
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
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="PYthon Neural Analysis Package Pour Laboratoires d’Excellence",
    install_requires=requirements,
    license="GNU General Public License v3",
    # long_description='pynapple is a Python library for analysing neurophysiological data. It allows to handle time series and epochs but also to use generic functions for neuroscience such as tuning curves and cross-correlogram of spikes. It is heavily based on neuroseries.' 
    # + '\n\n' + history,
    long_description=readme,
    include_package_data=True,
    keywords='neuroscience',
    name='pynapple',
    packages=find_packages(include=['pynapple', 'pynapple.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/PeyracheLab/pynapple',
    version='v0.3.1',
    zip_safe=False,
    long_description_content_type='text/markdown',
    download_url='https://github.com/PeyracheLab/pynapple/archive/refs/tags/v0.3.1.tar.gz'
)
