from setuptools import find_packages
from setuptools import setup

# TODO: confirm install requires
install_requires = [
                    'six >= 1.10.0',
                    'tensorflow==2.0.0-beta0',
                    'tfp-nightly',
                    'gym',
                    'matplotlib',
                    'tensorboardX',
                    'tabulate',
                    'protobuf',
                    'joblib',
                    ]

extras = dict()
extras['dev'] = [
    # Please keep alphabetized
    'coverage',
    'flake8',
    'flake8-docstrings',
    'flake8-import-order',
    'pep8-naming',
    'pre-commit',
    'pylint',
    'pytest>=3.6',  # Required for pytest-cov on Python 3.6
    'pytest-cov',
    'sphinx',
    'recommonmark',
    'yapf',
]

with open('README.md') as f:
    readme = f.read()

with open('VERSION') as v:
    version = v.read().strip()

setup(
    name='malib',
    version=version,
    include_package_data=True,
    packages=find_packages(exclude=['docs']),  # Required
    install_requires=install_requires,
    extras_require=extras,
    description='MALib: A FrameWork for Multi-Agent Learning',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
        'Development Status :: 0 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

    ],
    keywords='multi-agent reinforcement learning'
)
