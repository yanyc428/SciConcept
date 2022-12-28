# -*- coding: utf-8 -*-
from gensim.models import word2vec
import jieba
from tqdm import tqdm


class DataSet:
    def __init__(self, df):
        self.df = df
        self.pbar = tqdm(total=len(self) * 6, desc='Training W2v embeddings')

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        for index, row in self.df.iterrows():
            self.pbar.update(1)
            yield jieba.lcut(row.ABSTRACT)

    def __del__(self):
        self.pbar.close()


class W2V:
    def __init__(self, path, train_csv=None):
        self.path = path
        if train_csv is not None:
            self._train(DataSet(train_csv))
        else:
            self._load()

    def _load(self):
        self.model = word2vec.Word2Vec.load(self.path)

    def _train(self, it):
        self.model = word2vec.Word2Vec(it, min_count=1, vector_size=128)
        self.model.save(self.path)

    def similarity(self, word1, word2):
        try:
            return self.model.wv.similarity(word1, word2)
        except KeyError:
            return 0

    def most_similar(self, word):
        return self.model.wv.similar_by_word(word, topn=20)


if __name__ == '__main__':
    w2v = W2V('w2v.ckpt')
    print(w2v.model.wv.similarity('光纤通信', '用户程序'))
    print(w2v.most_similar('光纤通信'))
