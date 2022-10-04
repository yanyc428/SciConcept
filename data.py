import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import csv


class Dataset:
    def __init__(self):
        self.data = pd.read_csv('data.csv')
        self.keywords = []
        self.sorted_keywords = []
        with open('stopwords.txt', 'r') as f:
            self.stopwords = set([line.strip() for line in f.readlines()])
        self.onehot_vectors = dict()
        self.adjacency_matrix = None
        self.total_words = 10000
        self.id2word = {}
        self.word2id = {}
        self.extract_all_keywords()
        self.filter_keywords()

    def extract_all_keywords(self):
        for index, row in tqdm(self.data.iterrows(), total=len(self.data)):
            for word in row.AUTHOR_KEYWORDS.split('%'):
                if word not in self.stopwords:
                    self.keywords.append(word)

    def filter_keywords(self):
        counter = Counter(self.keywords)
        self.keywords = list(
            map(
                lambda x: x[0],
                sorted(dict(counter).items(), key=lambda x: (-x[1], x[0]))[:self.total_words]
            )
        )
        for index, word in enumerate(self.keywords):
            self.word2id[word] = index
            self.id2word[index] = word

    def sort_keywords(self):
        self.keywords = sorted(self.keywords, key=lambda x: self.calculate_adjacency_degree(x), reverse=True)

    def generate_adjacency_matrix(self):
        self.adjacency_matrix = np.zeros((self.total_words, self.total_words), dtype='int')
        for i in tqdm(range(self.total_words)):
            for j in range(self.total_words):
                self.adjacency_matrix[i, j] = self.calculate_co_occurrence(self.id2word[i], self.id2word[j])

    def calculate_adjacency_degree(self, word, with_weight=False):
        if word not in self.keywords:
            return -1
        index = self.word2id[word]
        if with_weight:
            return np.sum(self.adjacency_matrix[index])
        else:
            return np.sum(self.adjacency_matrix[index] > 0)

    def is_co_occur(self, word1, word2):
        return self.calculate_co_occurrence(word1, word2) != 0

    def generate_all_onehot_vectors(self):
        for word in tqdm(self.keywords):
            self.onehot_vectors[word] = self.generate_one_hot_vector(word)

    def calculate_co_occurrence(self, word1, word2):
        vec1 = self.onehot_vectors[word1]
        vec2 = self.onehot_vectors[word2]
        return np.bitwise_and(vec1, vec2).sum()

    def get_frequency(self, keyword):
        return self.data.ABSTRACT.str.contains(keyword, regex=False).sum()

    def generate_one_hot_vector(self, keyword):
        return np.array(self.data.ABSTRACT.str.contains(keyword, regex=False).astype('int').tolist())

    def export_adjacency_matrix(self):
        with open('result/adjacency_matrix.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow([''] + self.keywords)
            for word in self.keywords:
                index = self.word2id[word]
                writer.writerow([word] + self.adjacency_matrix[index].tolist())


if __name__ == '__main__':
    data = Dataset()
    data.generate_all_onehot_vectors()
    data.generate_adjacency_matrix()
    data.export_adjacency_matrix()
    print(data.calculate_adjacency_degree('数据挖掘'))
    print(data.calculate_adjacency_degree('数据挖掘', with_weight=True))
    data.sort_keywords()
