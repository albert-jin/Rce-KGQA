import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MetaQADataSet(Dataset):
    def __init__(self, entity_embed_path, entity_dict_path, relation_embed_path, relation_dict_path, qa_dataset_path,
                 split):
        """
            create MetaQADataSet
        :param entity_embed_path:
        :param entity_dict_path: filepath for mapping entity to index
        :param relation_embed_path:
        :param relation_dict_path:  filepath for mapping relation to index
        :param qa_dataset_path:
        :param split:
        """
        # ====load entity & relation embeddings====
        self.entity_embeddings = np.load(entity_embed_path)
        self.relation_embeddings = np.load(relation_embed_path)
        # ====load entity & relation dict (mapping word into index)====
        self.entities2idx = dict()
        self.relations2idx = dict()
        with open(entity_dict_path, 'rt', encoding='utf-8') as e_d, open(relation_dict_path, 'rt',
                                                                         encoding='utf-8') as r_d:
            for line in e_d:
                mapping = line.strip().split('\t')
                self.entities2idx[mapping[0]] = int(mapping[1])
            for line in r_d:
                mapping = line.strip().split('\t')
                self.relations2idx[mapping[0]] = int(mapping[1])
        self.entities_count = len(self.entities2idx)
        self.relations_count = len(self.relations2idx)

        # ====load QA pairs====
        self.results = []
        with open(qa_dataset_path, 'rt', encoding='utf-8') as inp_:
            for line in inp_.readlines():
                try:
                    line = line.strip().split('\t')
                    question = line[0]
                    q_temp = question.split('[')
                    topic_entity = q_temp[1].split(']')[0]
                    question = q_temp[0] + 'NE' + q_temp[1].split(']')[1]
                    answers = [i.strip() for i in line[1].split('|')]
                    self.results.append([topic_entity.strip(), question.strip(), answers])
                except RuntimeError:
                    continue
        assert len(self.results) > 0, f'read no qa-pairs in file [{qa_dataset_path}]'
        if split:
            split_result = []
            for qa_pair in self.results:
                for answer in qa_pair[2]:
                    split_result.append([qa_pair[0], qa_pair[1], answer])
            self.results = split_result
        # ====get word <=> idx mapping====
        self.idx_word = dict()
        self.word_idx = dict()
        self.max_sent_length = 0
        for qa_pair in self.results:  # retrieval all questions
            words = qa_pair[1].split()
            if len(words) > self.max_sent_length:
                self.max_sent_length = len(words)
            for word in words:
                if word not in self.word_idx:
                    self.word_idx[word] = len(self.word_idx)
                    self.idx_word[len(self.idx_word)] = word

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        qa_pair = self.results[index]
        # ==head entity text==
        text_head = qa_pair[0]
        head_idx = self.entities2idx[text_head]
        # ==text question==
        text_q = qa_pair[1]
        idx_q = [self.word_idx[word] for word in text_q.split()]
        # ==tail entity text==
        text_tails = qa_pair[2]
        idx_tails = [self.entities2idx[tail_text] for tail_text in text_tails]
        onehot_tail = torch.zeros(self.entities_count)
        onehot_tail.scatter_(0, torch.tensor(idx_tails), 1)
        return idx_q, head_idx, onehot_tail


class MetaQADataLoader(DataLoader):
    def __init__(self, entity_embed_path, entity_dict_path, relation_embed_path, relation_dict_path, qa_dataset_path,
                 batch_size=128, split=False, shuffle=True):
        dataset = MetaQADataSet(entity_embed_path, entity_dict_path, relation_embed_path, relation_dict_path,
                                qa_dataset_path, split)
        super(MetaQADataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                               collate_fn=self._collate_fn)

    @staticmethod
    def _collate_fn(batch_data):
        """
        :param batch_data: dataset __getitem__ outputs.
        :return: batch_questions_index, batch_questions_length, batch_head_entity, batch_onehot_answers, max_sent_len
        """
        sorted_qa_pairs = list(sorted(batch_data, key=lambda x: len(x[0]), reverse=True))
        sorted_qa_pairs_len = [len(qa_pair[0]) for qa_pair in sorted_qa_pairs]
        max_sent_len = len(sorted_qa_pairs[0][0])
        padded_questions = []  # torch.zeros(batch_size, max_sent_len, dtype=torch.long)
        head_idxs = []
        onehot_tails = []
        for idx_q, head_idx, onehot_tail in sorted_qa_pairs:
            padded_questions.append(idx_q + [0] * (max_sent_len - len(idx_q)))
            head_idxs.append(head_idx)
            onehot_tails.append(onehot_tail)
        return torch.tensor(padded_questions, dtype=torch.long), torch.tensor(sorted_qa_pairs_len, dtype=torch.long), \
               torch.tensor(head_idxs), torch.tensor(onehot_tails), max_sent_len
