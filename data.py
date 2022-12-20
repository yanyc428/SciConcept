import os

import click
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import csv
import pickle
import jieba
from embeddings import W2V
from database import Dataset as DaS, Session
import datetime
import shutil


class Dataset:
    def __init__(self, ckpt, mode='w'):
        keywords_file = os.path.join('checkpoints', ckpt, 'keywords.txt')
        data_file = os.path.join('checkpoints', ckpt, 'data.csv')
        self.data = pd.read_csv(data_file).astype('str').drop_duplicates()
        self.data.ABSTRACT = self.data.ABSTRACT.str.upper()
        with open('stopwords.txt', 'r') as f:
            self.stopwords = set([line.strip() for line in f.readlines()])

        if mode == 'r':
            with open(os.path.join('checkpoints', ckpt, 'keywords.pkl'), 'rb') as f:
                self.keywords = pickle.load(f)
            with open(os.path.join('checkpoints', ckpt, 'semantic_matrix.pkl'), 'rb') as f:
                self.semantic_matrix = pickle.load(f)
            with open(os.path.join('checkpoints', ckpt, 'adjacency_matrix.pkl'), 'rb') as f:
                self.adjacency_matrix = pickle.load(f)
            with open(os.path.join('checkpoints', ckpt, 'total_words.pkl'), 'rb') as f:
                self.total_words = pickle.load(f)
            with open(os.path.join('checkpoints', ckpt, 'id2word.pkl'), 'rb') as f:
                self.id2word = pickle.load(f)
            with open(os.path.join('checkpoints', ckpt, 'word2id.pkl'), 'rb') as f:
                self.word2id = pickle.load(f)
            with open(os.path.join('checkpoints', ckpt, 'invert_index.pkl'), 'rb') as f:
                self.invert_index = pickle.load(f)
            self.w2v = W2V(os.path.join('checkpoints', ckpt, 'w2v.ckpt'))

        elif mode == 'w':
            self.session = Session()
            self.das = self.session.query(DaS).filter_by(name=ckpt).first()
            self.das.papers = len(self.data)
            self.session.commit()

            self.keywords = set()
            self.total_words = 10000
            self.das.status = '正在处理关键词'
            self.session.commit()
            if os.path.exists(keywords_file):
                self._load_keywords_from_file(keywords_file)
            else:
                self.data.AUTHOR_KEYWORDS = self.data.AUTHOR_KEYWORDS.str.upper()
                self._extract_all_keywords()
            self.das.keywords = len(self.keywords)
            self.session.commit()

            for word in self.keywords:
                jieba.add_word(word)
            self.invert_index = {}
            self._generate_invert_index()

            self._filter_keywords()
            self.total_words = len(self.keywords)
            self.das.real_keywords = len(self.keywords)
            self.session.commit()

            self.das.status = '正在训练词向量'
            self.session.commit()
            self.w2v = W2V(os.path.join('checkpoints', ckpt, 'w2v.ckpt'), self.data)

            self.id2word = {}
            self.word2id = {}
            self._generate_mapping_dicts()

            self.adjacency_matrix = None
            self.semantic_matrix = None
            self._generate_adjacency_matrix()
            self.export_adjacency_matrix()
            self.export_adjacency_matrix(semantic=True)

            with open(os.path.join('checkpoints', ckpt, 'keywords.pkl'), 'wb') as f:
                pickle.dump(self.keywords, f)
            with open(os.path.join('checkpoints', ckpt, 'adjacency_matrix.pkl'), 'wb') as f:
                pickle.dump(self.adjacency_matrix, f)
            with open(os.path.join('checkpoints', ckpt, 'semantic_matrix.pkl'), 'wb') as f:
                pickle.dump(self.semantic_matrix, f)
            with open(os.path.join('checkpoints', ckpt, 'total_words.pkl'), 'wb') as f:
                pickle.dump(self.total_words, f)
            with open(os.path.join('checkpoints', ckpt, 'id2word.pkl'), 'wb') as f:
                pickle.dump(self.id2word, f)
            with open(os.path.join('checkpoints', ckpt, 'word2id.pkl'), 'wb') as f:
                pickle.dump(self.word2id, f)
            with open(os.path.join('checkpoints', ckpt, 'invert_index.pkl'), 'wb') as f:
                pickle.dump(self.invert_index, f)
            self.das.status = '导入完成'
            self.session.commit()

    def _extract_all_keywords(self):
        keywords = []
        for index, row in tqdm(self.data.iterrows(), total=len(self.data), desc='extract_all_keywords'):
            for word in row.AUTHOR_KEYWORDS.split('%'):
                keywords.append(word)
        counter = Counter(keywords)
        self.keywords = set(
            map(
                lambda x: x[0],
                sorted(dict(counter).items(), key=lambda x: (-x[1], x[0]))[:self.total_words]
            )
        )

    def _load_keywords_from_file(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            self.das.keywords = len(lines)
            self.session.commit()
            for line in tqdm(lines, desc='loading keywords from file'):
                word = line.strip().upper()
                self.keywords.add(word)

    def _filter_keywords(self, min_length=1):
        for keyword in self.keywords.copy():
            if keyword not in self.invert_index:
                self.keywords.remove(keyword)
            elif keyword in self.stopwords:
                self.keywords.remove(keyword)
            elif keyword.encode("utf-8").isalpha():
                self.keywords.remove(keyword)
            elif len(keyword) <= min_length:
                self.keywords.remove(keyword)
            elif len(self.keywords) >= self.total_words:
                self.keywords.remove(keyword)

    def _generate_mapping_dicts(self):
        for index, word in enumerate(self.keywords):
            self.word2id[word] = index
            self.id2word[index] = word

    def _generate_adjacency_matrix(self):
        self.das.status = '正在生成邻接矩阵'
        self.session.commit()
        self.adjacency_matrix = np.zeros((self.total_words, self.total_words), dtype='int')
        self.semantic_matrix = np.zeros((self.total_words, self.total_words), dtype='float32')
        for i in tqdm(range(self.total_words), desc='generate_adjacency_matrix'):
            for j in range(self.total_words):
                self.adjacency_matrix[i, j] = self._calculate_co_occurrence_by_invert_index(self.id2word[i],
                                                                                            self.id2word[j])
                self.semantic_matrix[i, j] = self.w2v.similarity(self.id2word[i], self.id2word[j])

    def calculate_adjacency_degree(self, word, with_weight=False, semantic=False):
        if word not in self.keywords:
            return -1
        index = self.word2id[word]
        if semantic:
            return np.sum(self.semantic_matrix[index]) - self.get_frequency(word, semantic=True)
        if with_weight:
            return np.sum(self.adjacency_matrix[index]) - self.get_frequency(word, semantic=False)
        else:
            return np.sum(self.adjacency_matrix[index] > 0) - 1

    def _generate_invert_index(self):
        self.das.status = '正在生成倒排索引'
        self.session.commit()
        for index, row in tqdm(self.data.iterrows(), desc='generate_invert_index', total=len(self.data)):
            tokens = jieba.cut(row.ABSTRACT)
            for token in tokens:
                if token not in self.invert_index:
                    self.invert_index[token] = set()
                self.invert_index[token].add(index)

    def _calculate_co_occurrence_by_invert_index(self, word1, word2):
        set1 = self.invert_index[word1]
        set2 = self.invert_index[word2]
        return len(set1 & set2)

    def get_frequency(self, keyword, semantic=False):
        if keyword not in self.keywords:
            return -1
        i = self.word2id[keyword]
        if semantic:
            return self.semantic_matrix[i, i]
        return self.adjacency_matrix[i, i]

    def get_co_occurrence(self, word1, word2):
        i = self.word2id[word1]
        j = self.word2id[word2]
        return self.adjacency_matrix[i, j]

    def get_semantic_similarity(self, word1, word2):
        i = self.word2id[word1]
        j = self.word2id[word2]
        return self.semantic_matrix[i, j]

    def export_adjacency_matrix(self, semantic=False):
        if semantic:
            filename = 'result/semantic_matrix.csv'
        else:
            filename = 'result/adjacency_matrix.csv'
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([''] + list(self.keywords))
            for word in tqdm(self.keywords, desc='export adjacency matrix:'):
                index = self.word2id[word]
                if semantic:
                    writer.writerow([word] + self.semantic_matrix[index].tolist())
                else:
                    writer.writerow([word] + self.adjacency_matrix[index].tolist())


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """导入数据集"""
    pass


@cli.command(name='list', help='显示已导入的数据集', context_settings=CONTEXT_SETTINGS)
def _list():
    session = Session()
    datasets = session.query(DaS).filter_by(status='导入完成').all()
    session.close()
    print(
        f'{"":4s} {"Name":20s} {"Date":10s} {"Papers":10s} {"Keywords":10s} {"Valid_Keywords":14s} {"Status":10s}')
    for index, item in enumerate(datasets):
        print(
            f'{index + 1:4d} {item.name:20s} {str(item.create_date):10s} {item.papers:<10d} {item.keywords:<10d} {item.real_keywords:<14d} {item.status:10s}')


@cli.command(help='添加数据集', context_settings=CONTEXT_SETTINGS)
@click.option('--name', '-n', required=True, type=str, help='数据集名称')
@click.option('--data', '-d',required=True, type=str, help='文档文件的路径，csv格式，至少包含ABSTRACT字段，若不实用自建术语表，还需要提供AUTHOR_KEYWORDS字段')
@click.option('--keywords', '-k', type=str, help='自建术语表的路径，txt格式，每行一个词语', default=None)
def add(name, data, keywords):
    if not os.path.exists(os.path.join('checkpoints', name)):
        os.makedirs(os.path.join('checkpoints', name))

    shutil.copy(data, os.path.join('checkpoints', name, 'data.csv'))
    if keywords is not None:
        shutil.copy(keywords, os.path.join('checkpoints', name, 'keywords.txt'))

    dataset = DaS(name=name, papers=0, keywords=0, real_keywords=0, create_date=datetime.datetime.now(),
                  status='正在创建')
    session = Session()
    session.add(dataset)
    session.commit()
    session.close()

    try:
        Dataset(name, mode='w')
    except Exception as e:
        print(e)
        session = Session()
        ds = session.query(DaS).filter_by(name=args.ckpt).first()
        ds.status = '导入失败'
        session.commit()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt')

    args = parser.parse_args()
    try:
        Dataset(args.ckpt, mode='w')
    except Exception as e:
        print(e)
        session = Session()
        ds = session.query(DaS).filter_by(name=args.ckpt).first()
        ds.status = '导入失败'
        session.commit()
