import click

from data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


class ConceptTree:
    def __init__(self, dataset, k):
        self.dataset = Dataset(ckpt=dataset, mode='r')
        self.unallocated_words = self.dataset.keywords.copy()
        self.root = ConceptNode('root', tree=self)
        self.k = k

    def visualize(self):
        self.root.visualize()

    def search(self, keyword):
        res = self.root.search(keyword, [])
        # if res:
        #     return res
        # keyword = self.dataset.w2v.most_similar(keyword)
        return res

    def construct(self):
        self.pbar = tqdm(total=len(self.unallocated_words))
        self.root.set_sub_words(self.unallocated_words)
        self.root.generate_children(self.k)
        self.root.allocate_sub_words(self.k)
        self.pbar.close()
        print('Success!')
        self.pbar = None

    def export(self):
        return self.root.export()


class ConceptNode:
    def __init__(self, word, tree, level=0):
        self.word = word
        self.tree = tree
        self.level = level

        self.children = []
        self.sub_words = []
        self.semantic_scores = {}
        self.topology_scores = {}

    def set_sub_words(self, words):
        self.sub_words = words

    def generate_children(self, k):
        while True:
            self._update_scores()  # 更新权重
            sorted_sub_words = sorted(
                self.sub_words,
                key=lambda x: max(self.semantic_scores[x], self.topology_scores[x]),  # 取拓扑特征和语义特征中更大的那个
                reverse=True
            )
            for child in self.children:
                sorted_sub_words.remove(child.word)
            if len(sorted_sub_words) == 0:
                break
            if len(self.children) == k:
                break
            else:
                self.children.append(ConceptNode(sorted_sub_words[0], self.tree, self.level + 1))

    def _update_scores(self, punish=True, consider_duplicate=True):
        for word in self.sub_words:
            self.topology_scores[word] = self.tree.dataset.calculate_adjacency_degree(word, )
            self.semantic_scores[word] = self.tree.dataset.calculate_adjacency_degree(word, semantic=True)
            if punish:
                for child in self.children:
                    self.topology_scores[word] -= self.tree.dataset.get_co_occurrence(word, child.word)
                    self.semantic_scores[word] -= self.tree.dataset.get_semantic_similarity(word, child.word)
            if consider_duplicate:
                duplicate = len(set(self.word) & set(word)) + 1
                self.topology_scores[word] *= duplicate
        self._scale_scores()

    def _scale_scores(self):
        if len(self.sub_words) > 0:
            scaler = StandardScaler()
            scaler.fit(np.array(list(self.semantic_scores.values())).reshape(-1, 1))
            for word in self.semantic_scores:
                self.semantic_scores[word] = scaler.transform([[self.semantic_scores[word]]])[0, 0]
            scaler.fit(np.array(list(self.topology_scores.values())).reshape(-1, 1))
            for word in self.topology_scores:
                self.topology_scores[word] = scaler.transform([[self.topology_scores[word]]])[0, 0]

    def allocate_sub_words(self, k):
        sub_words = {}

        for child in self.children:
            sub_words[child.word] = []

        for word in self.sub_words:
            if word in sub_words:
                continue
            co_occur = []
            for child in self.children:
                co_occur.append(self.tree.dataset.get_co_occurrence(word, child.word))
            if len(co_occur) == 0:
                return
            # if max(co_occur) == 0:
            #     continue
            if max(co_occur) == 0:
                co_occur = []
                for child in self.children:
                    co_occur.append(self.tree.dataset.get_semantic_similarity(word, child.word))
            index = co_occur.index(max(co_occur))
            sub_words[self.children[index].word].append(word)

        for child in self.children:
            child.set_sub_words(sub_words[child.word])
            child.generate_children(k)
            child.allocate_sub_words(k)
            self.tree.pbar.update(1)

    def visualize(self):
        print('\t' * self.level + '{}'.format(self.word))
        if len(self.children) > 0:
            for child in self.children:
                child.visualize()

    def search(self, keyword, path_list, path=()):
        path += (self.word, )

        if keyword in self.word:
            path_list.append(path)

        for child in self.children:
            child.search(keyword, path_list, path)

        return path_list

    def export(self):
        return {'label': self.word, 'children': [child.export() for child in self.children]}


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """术语规范化"""
    pass


@cli.command(name='generate', help='生成层次分类体系', context_settings=CONTEXT_SETTINGS)
@click.option('--name', '-n', help='数据集名称，可使用datasets命令查看所有数据集', type=str)
@click.option('-k', help='层级最大节点数', type=int, default=15)
def generate(name, k):
    tree = ConceptTree(name, k)
    tree.construct()
    with open('tree.tmp', 'wb') as f:
        pickle.dump(tree, f)


@cli.command(name='search', help='术语规范化', context_settings=CONTEXT_SETTINGS)
@click.argument('word', type=str)
@click.option('--semantic', '-s', help='使用语义检索', type=int, default=None)
def search(word, semantic):
    with open('tree.tmp', 'rb') as f:
        tree = pickle.load(f)
    if semantic:
        i = 1
        try:
            if word in tree.dataset.keywords:
                print('Path containing {}:'.format(word))
                for index, item in enumerate(tree.search(word)):
                    # item = [i.replace(word, f'*{word}*') for i in item]
                    print('{}. {}'.format(index + 1, ' -> '.join(item)))

            for keyword, _ in tree.dataset.w2v.most_similar(word):
                print()
                if keyword in tree.dataset.keywords:
                    word = keyword
                    print(f'Similar word {i}: ', word)
                    for index, item in enumerate(tree.search(word)):
                        # item = [i.replace(word, f'*{word}*') for i in item]
                        print('{}. {}'.format(index + 1, ' -> '.join(item)))
                    i += 1
                    if i > semantic:
                        return
        except KeyError:
            print('Word not in glossary:', word)
            return
    else:
        print('Path containing {}:'.format(word))
        for index, item in enumerate(tree.search(word)):
            # item = [i.replace(word, f'*{word}*') for i in item]
            print('{}. {}'.format(index+1, ' -> '.join(item)))


if __name__ == '__main__':
    tree = ConceptTree('dataset_6.5K_7K', k=15)
    tree.construct()
    # tree.visualize()
    print(tree.search('VR'))
    # data = Dataset(ckpt='ckpt3', mode='r')
    # pbar = tqdm(total=len(dataset.keywords), desc='generate_concept_nodes')
    # root = ConceptNode('root')
    # root.set_sub_words(dataset.keywords)
    #
    # root.visualize()
    # print(root.search('需求变更'))
    # pbar.close()
