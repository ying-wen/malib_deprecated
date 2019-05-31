import codecs
from os import path
from setuptools import find_packages
from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

# TODO: confirm install requires
install_requires = [
                    'six >= 1.10.0',
                    'tensorflow==2.0.0-alpha0',
                    'tfp-nightly',
                    'gym',
                    'matplotlib',
                    'tensorboardX',
                    'tabulate',
                    'protobuf',
                    'joblib',
                    ]
test_requirements = ['six >= 1.10.0', 'absl-py >= 0.1.6']

setup(
    name='malib',
    version='0.0.1',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),  # Required
    install_requires=install_requires,
    extras_require={},
    tests_require=test_requirements,
    description='malib: A FrameWork for Multi-Agent Deep Reinforcement Learning',
    long_description=long_description,
    classifiers=[  # Optional
        'Development Status :: 0 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

    ],
    license='MIT',
    keywords='multi-agent reinforcement learning'
)
