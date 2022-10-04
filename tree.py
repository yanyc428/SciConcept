from data import Dataset
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

    def generate_children(self, k=15):
        for word in self.sub_words[:k]:
            pbar.update(1)
            self.children.append(ConceptNode(word, self.level + 1))

    def allocate_sub_words(self):
        sub_words = {}

        for child in self.children:
            sub_words[child.word] = []

        for word in self.sub_words:
            if word in sub_words:
                continue
            co_occur = []
            for child in self.children:
                co_occur.append(dataset.calculate_co_occurrence_by_invert_index(word, child.word))
            index = co_occur.index(max(co_occur))
            sub_words[self.children[index].word].append(word)

        for child in self.children:
            child.set_sub_words(sub_words[child.word])
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
    pbar = tqdm(total=len(dataset.keywords), desc='generate_concept_nodes')
    root = ConceptNode('root')
    root.set_sub_words(dataset.keywords)
    root.generate_children()
    root.allocate_sub_words()
    root.visualize()
    pbar.close()
