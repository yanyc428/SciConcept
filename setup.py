# -*- coding: utf-8 -*-
"""
@Project    : SciConcept
@File       : setup
@Email      : yanyuchen@zju.edu.cn
@Author     : Yan Yuchen
@Time       : 2022/12/20 21:59
"""
from setuptools import setup

setup(
    name='sci-concept',
    version='0.1.0',
    py_modules=['command', 'data', 'database', 'embeddings', 'tree', 'log'],
    install_requires=[
        'Click',
        'tqdm',
        'numpy',
        'pandas',
        'sqlalchemy'
    ],
    entry_points={
        'console_scripts': [
            'sci-concept = command:main',
            'sci-concept-datasets = data:cli',
            'sci-concept-tree = tree:cli',
        ],
    },
)
