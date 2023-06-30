"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li and Moishe Keselman.
"""

# Build the OpenLabCluster as PyPI package
VERSION = '0.0.36'
DESCRIPTION = 'OpenLabCluster'
LONG_DESCRIPTION = open('README.md', 'r').read()
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openlabcluster",
    version=VERSION,
    author="Jingyuan Li",
    author_email="jingyli6@uw.edu",
    description="OpenLabCluster",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/shlizee/OpenLabCluster",
    install_requires=REQUIREMENTS,
    extras_require={
        "user_interface": ["wxpython<4.1"],
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    keywords=['python', 'first package'],
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    entry_points='''
        [gui_scripts]
        openlabcluster=openlabcluster.user_interface.run_gui_pythonw:run
    ''',
)