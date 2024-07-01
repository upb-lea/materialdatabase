#!/usr/bin/env python
"""The setup script."""
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

# with open('requirements.txt', 'r') as f:
#    requirements = f.read().splitlines()

setup_requirements = ['setuptools_scm']

test_requirements = ['pytest']

setup(
    author="LEA - Uni Paderborn",
    author_email='upblea@mail.upb.de',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Environment :: MacOS X'
    ],
    description="materialdatabase",
    install_requires=['numpy~=1.26.0',
                      'scipy>=1.6.0',
                      'setuptools>=49.2.1',
                      'matplotlib>=3.3.4',
                      'pytest>=6.2.4',
                      'mplcursors',
                      'deepdiff'],
    license="GNU General Public License v3",

    # long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='materialdatabase',
    name='materialdatabase',
    packages=find_packages(include=['materialdatabase', 'materialdatabase.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={},
    url='https://github.com/upb-lea/materialdatabase',
    project_urls={
        "Documentation": "https://upb-lea.github.io/materialdatabase/",
        "Source Code": "https://github.com/upb-lea/materialdatabase",
    },
    version='0.3.0',
    zip_safe=False,
    data_files=[('', ['CHANGELOG.md'])]
)
