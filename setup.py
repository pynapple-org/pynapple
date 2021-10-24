from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'PYthon Neural Analysis Package Pour Laboratoires d’Excellence'
LONG_DESCRIPTION = '''
Python package for analysing neurophysiological and behavioral data
'''

# Setting up
setup(       
        name="pynapple", 
        version=VERSION,
        author="Guillaume Viejo",
        author_email="guillaume.viejo@gmail.com ",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'electrophysiology', 'time-series'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Researchers",
            "Programming Language :: Python :: 3",
            # "Operating System :: MacOS :: MacOS X",
            # "Operating System :: Microsoft :: Windows",
        ]
)   