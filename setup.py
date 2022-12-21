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
    py_modules=['commands', 'data', 'database', 'embeddings', 'tree'],
    install_requires=[
        'Click',
        'tqdm',
        'numpy',
        'pandas',
        'sqlalchemy',
        'jieba',
        'gensim',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'sci-concept = commands:main',
            'sci-concept-datasets = data:cli',
            'sci-concept-tree = tree:cli',
        ],
    },
)
