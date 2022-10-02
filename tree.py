from data import Dataset
import numpy as np
from tqdm import tqdm
from log import enable_global_logging_config, logger as logging


class ConceptNode:
    def __init__(self, word, level=0):
        self.word = word
        self.children = []
        self.sub_words = []
        self.level = level

    def set_sub_words(self, words):
        self.sub_words = words

    def generate_children(self, k=10):
        words_with_weights = []
        for word in tqdm(self.sub_words):
            words_with_weights.append((-sum(dataset.generate_one_hot_vector(word)), word))
        for weight, word in sorted(words_with_weights)[:k]:
            self.children.append(ConceptNode(word, self.level + 1))

    def allocate_sub_words(self):
        sub_words = {}
        mat = []
        for child in self.children:
            sub_words[child.word] = []
            mat.append(dataset.generate_one_hot_vector(child.word))
        mat = np.array(mat)

        for word in self.sub_words:
            if word in sub_words:
                continue
            vec = dataset.generate_one_hot_vector(word)
            temp = np.matmul(mat, vec)
            index = np.argmax(temp)
            sub_words[self.children[index].word].append(word)

        for child in self.children:
            child.set_sub_words(sub_words[child.word])

        if self.level < 3:
            for child in self.children:
                child.generate_children()
                child.allocate_sub_words()

    def visualize(self):
        logging.info('\t' * self.level + '{}'.format(self.word))
        if len(self.children) > 0:
            for child in self.children:
                child.visualize()


if __name__ == '__main__':
    enable_global_logging_config()
    dataset = Dataset()
    root = ConceptNode('root')
    root.set_sub_words(dataset.keywords)
    root.generate_children()
    root.allocate_sub_words()
    root.visualize()
