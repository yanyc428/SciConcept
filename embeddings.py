# -*- coding: utf-8 -*-
"""
@Project    : SciConcept
@File       : embeddings
@Email      : yanyuchen@zju.edu.cn
@Author     : Yan Yuchen
@Time       : 2022/12/12 20:59
"""
from gensim.models import word2vec
import jieba
from tqdm import tqdm
jieba.load_userdict('keywords.txt')


class DataSet:
    def __init__(self, df, keywords):
        self.df = df
        self.pbar = tqdm(total=len(self) * 6, desc='Training W2v embeddings')
        self.keywords = keywords
        self.words = set()

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        for index, row in self.df.iterrows():
            self.pbar.update(1)
            yield jieba.lcut(row.ABSTRACT)

    def __del__(self):
        self.pbar.close()


class W2V:
    def __init__(self, train_csv=None, keywords=None):
        if train_csv is not None:
            self._train(DataSet(train_csv, keywords))
        else:
            self._load()

    def _load(self):
        self.model = word2vec.Word2Vec.load('w2v.ckpt')

    def _train(self, it):
        self.model = word2vec.Word2Vec(it, min_count=1, vector_size=128)
        self.model.save('w2v.ckpt')

    def similarity(self, word1, word2):
        try:
            return self.model.wv.similarity(word1, word2)
        except KeyError:
            return 0

    def most_similar(self, word):
        return self.model.wv.similar_by_word(word, topn=1)


if __name__ == '__main__':
    w2v = W2V()
    print(w2v.model.wv.similarity('光纤通信', '用户程序'))
    print(w2v.most_similar('光纤通信'))
