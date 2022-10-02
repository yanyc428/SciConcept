import pandas as pd
import numpy as np
from tqdm import tqdm


class Dataset:
    def __init__(self):
        self.data = pd.read_csv('data.csv').sample(1000)
        self.keywords = set()
        with open('stopwords.txt', 'r') as f:
            self.stopwords = set([line.strip() for line in f.readlines()])
        self.extract_all_keywords()

    def extract_all_keywords(self):
        for index, row in tqdm(self.data.iterrows(), total=len(self.data)):
            for word in row.AUTHOR_KEYWORDS.split('%'):
                if word not in self.stopwords:
                    self.keywords.add(word)
        self.keywords = sorted(self.keywords)

    def generate_one_hot_vector(self, keyword):
        return np.array(self.data.ABSTRACT.str.contains(keyword, regex=False).astype('int').tolist())


if __name__ == '__main__':
    data = Dataset()
    print(sum(data.generate_one_hot_vector('深度学习')))
    print(len(data.keywords))
