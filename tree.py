from data import Dataset
from tqdm import tqdm
from log import enable_global_logging_config, logger as logging


class ConceptNode:
    def __init__(self, word, level=0):
        self.word = word
        self.children = []
        self.sub_words = []
        self.level = level
        self.scores = {}

    def set_sub_words(self, words):
        self.sub_words = words
        self._sort_sub_words_by_degree()

    def _sort_sub_words_by_degree(self):
        self.sub_words = list(sorted(
            self.sub_words,
            key=lambda x: dataset.calculate_adjacency_degree(x),
            reverse=True
        ))

    def generate_children(self, k=15):
        while len(self.children) <= k:
            self.update_scores()
            sorted_sub_words = sorted(self.sub_words, key=lambda x: self.scores[x], reverse=True)

            for child in self.children:
                sorted_sub_words.remove(child.word)

            if len(sorted_sub_words) == 0:
                return

            self.children.append(ConceptNode(sorted_sub_words[0], self.level + 1))
            pbar.update(1)

    def update_scores(self):
        for word in self.sub_words:
            self.scores[word] = dataset.calculate_adjacency_degree(word, with_weight=True)
            for child in self.children:
                self.scores[word] -= dataset.get_co_occurrence(word, child.word)
            duplicate = len(set(self.word) & set(word)) + 1
            self.scores[word] *= duplicate

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
