import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import csv
import pickle


class Dataset:
    def __init__(self, file=None, mode='a'):
        self.data = pd.read_csv('data.csv').astype('str')
        with open('stopwords.txt', 'r') as f:
            self.stopwords = set([line.strip() for line in f.readlines()])

        if mode == 'r' or mode == 'a':
            with open('keywords.pkl', 'rb') as f:
                self.keywords = pickle.load(f)
            with open('adjacency_matrix.pkl', 'rb') as f:
                self.adjacency_matrix = pickle.load(f)
            with open('total_words.pkl', 'rb') as f:
                self.total_words = pickle.load(f)
            with open('id2word.pkl', 'rb') as f:
                self.id2word = pickle.load(f)
            with open('word2id.pkl', 'rb') as f:
                self.word2id = pickle.load(f)
            with open('invert_index.pkl', 'rb') as f:
                self.invert_index = pickle.load(f)
        elif mode == 'w':
            self.keywords = set()
            self.adjacency_matrix = None
            self.total_words = 1000
            self.id2word = {}
            self.word2id = {}
            self.invert_index = {}
            if file:
                self.load_keywords_from_file(file)
            else:
                self.extract_all_keywords()
            self.generate_mapping_dicts()
            self.generate_invert_index()
            self.generate_adjacency_matrix()
            self.export_adjacency_matrix()
            self.sort_keywords()

        if mode == 'w' or mode == 'a':
            with open('keywords.pkl', 'wb') as f:
                pickle.dump(self.keywords, f)
            with open('adjacency_matrix.pkl', 'wb') as f:
                pickle.dump(self.adjacency_matrix, f)
            with open('total_words.pkl', 'wb') as f:
                pickle.dump(self.total_words, f)
            with open('id2word.pkl', 'wb') as f:
                pickle.dump(self.id2word, f)
            with open('word2id.pkl', 'wb') as f:
                pickle.dump(self.word2id, f)
            with open('invert_index.pkl', 'wb') as f:
                pickle.dump(self.invert_index, f)

    def extract_all_keywords(self):
        keywords = []
        for index, row in tqdm(self.data.iterrows(), total=len(self.data), desc='extract_all_keywords'):
            for word in row.AUTHOR_KEYWORDS.split('%'):
                if word not in self.stopwords:
                    keywords.append(word)
        counter = Counter(keywords)
        self.keywords = set(
            map(
                lambda x: x[0],
                sorted(dict(counter).items(), key=lambda x: (-x[1], x[0]))[:self.total_words]
            )
        )

    def generate_mapping_dicts(self):
        for index, word in enumerate(self.keywords):
            self.word2id[word] = index
            self.id2word[index] = word

    def load_keywords_from_file(self, file):
        with open(file, 'r') as f:
            for line in f:
                word = line.strip()
                if word not in self.stopwords:
                    self.keywords.add(word)
                if len(self.keywords) >= self.total_words:
                    break

    def sort_keywords(self):
        self.keywords = sorted(self.keywords, key=lambda x: self.calculate_adjacency_degree(x), reverse=True)

    def generate_adjacency_matrix(self):
        self.adjacency_matrix = np.zeros((self.total_words, self.total_words), dtype='int')
        for i in tqdm(range(self.total_words), desc='generate_adjacency_matrix'):
            for j in range(self.total_words):
                self.adjacency_matrix[i, j] = self.calculate_co_occurrence_by_invert_index(self.id2word[i],
                                                                                           self.id2word[j])

    def calculate_adjacency_degree(self, word, with_weight=False):
        if word not in self.keywords:
            return -1
        index = self.word2id[word]
        if with_weight:
            return np.sum(self.adjacency_matrix[index]) - self.get_frequency(word)
        else:
            return np.sum(self.adjacency_matrix[index] > 0) - 1

    def generate_invert_index(self):
        doc_ids = np.arange(len(self.data))
        for word in tqdm(self.keywords, desc='generate_invert_index'):
            vec = self.generate_one_hot_vector(word)
            self.invert_index[word] = set(doc_ids[vec.astype('bool')])

    def calculate_co_occurrence_by_invert_index(self, word1, word2):
        set1 = self.invert_index[word1]
        set2 = self.invert_index[word2]
        return len(set1 & set2)

    def get_frequency(self, keyword):
        i = self.word2id[keyword]
        return self.adjacency_matrix[i, i]

    def get_co_occurrence(self, word1, word2):
        i = self.word2id[word1]
        j = self.word2id[word2]
        return self.adjacency_matrix[i, j]

    def generate_one_hot_vector(self, keyword):
        return np.array(self.data.ABSTRACT.str.contains(keyword, regex=False).astype('int'))

    def export_adjacency_matrix(self):
        with open('result/adjacency_matrix.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow([''] + list(self.keywords))
            for word in self.keywords:
                index = self.word2id[word]
                writer.writerow([word] + self.adjacency_matrix[index].tolist())


if __name__ == '__main__':
    data = Dataset()
    print(data.get_frequency('贝叶斯网络'))
    print(data.calculate_adjacency_degree('贝叶斯网络'))
    print(data.calculate_adjacency_degree('贝叶斯网络', with_weight=True))
