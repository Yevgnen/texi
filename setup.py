# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="texi",
    description="Text processing toolbox.",
    version="0.1.0",
    url="https://github.com/Yevgnen/texi",
    author="Yevgnen Koh",
    author_email="wherejoystarts@gmail.com",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    install_requires=[],
    test_suite="tests",
    zip_safe=False,
)
