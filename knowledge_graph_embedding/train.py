from collections import defaultdict
from model import ComplEx_KGE
import numpy as np
import os
import time
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class DataSet:

    def __init__(self, data_dir, reverse):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = sorted(list(set([d[0] for d in self.data] + [d[2] for d in self.data])))
        self.relations = sorted(list(set([d[1] for d in self.data])))

    @staticmethod
    def load_data(data_dir, data_type, reverse):
        with open(f"{data_dir}\\{data_type}.txt", 'r', encoding='utf-8') as inp:
            data = [line.strip().split('\t') for line in inp.readlines()]
            if reverse:
                data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        return data


def set_fixed_seed(seed):
    if not torch.cuda.is_available:
        print('Sorry, you should buy an NVIDIA Graphic Processing Unit and set CUDA environment!')
        exit(-1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Experiment:

    def __init__(self, input_dropout, hidden_dropout1, hidden_dropout2, learning_rate, ent_vec_dim, rel_vec_dim,
                 num_epochs, test_interval, batch_size, decay_rate, label_smoothing, dataset_name, data_dir,
                 model_dir, load_from, do_batch_norm):
        # =======hyper-parameters===========
        self.input_dropout = input_dropout
        self.hidden_dropout1 = hidden_dropout1
        self.hidden_dropout2 = hidden_dropout2
        self.do_batch_norm = do_batch_norm
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_epochs = num_epochs
        self.test_interval = test_interval
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        # ========dataset stored and loaded==========
        self.dataset_name = dataset_name
        # ====KG sources====
        # before you run next statement, you should make knowledge_graphs available in root directory.
        data_path = os.path.join(data_dir, self.dataset_name)
        assert os.path.isdir(data_path)
        self.dataset = DataSet(data_dir=data_path, reverse=True)
        self.entity_idxs = {entity: i for i, entity in enumerate(self.dataset.entities)}
        self.relation_idxs = {relation: i for i, relation in enumerate(self.dataset.relations)}
        # ====KGE model save=====
        model_save_dir = os.path.join(model_dir, self.dataset_name)
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)
        self.best_model_save_dir = os.path.join(model_save_dir, 'best_checkpoint')
        if not os.path.isdir(self.best_model_save_dir):
            os.mkdir(self.best_model_save_dir)
        self.final_model_save_dir = os.path.join(model_save_dir, 'final_checkpoint')
        if not os.path.isdir(self.final_model_save_dir):
            os.mkdir(self.final_model_save_dir)
        # ====从某一断点恢复训练====
        self.load_from = load_from

    def get_data_idxs(self, triples):
        """ 实体关系实体下标的三元组 [(234,13,424),..] """
        return [(self.entity_idxs[triple[0]], self.relation_idxs[triple[1]], self.entity_idxs[triple[2]]) for triple in triples]

    def get_hl_t(self, triples):  # h: head entity, l: relationship, t:tail entity.
        """ {(头实体下标,关系下标):[尾实体下标,尾实体下标...],others:[],...} """
        er_vocab = defaultdict(list)
        for triple in triples:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = torch.zeros([len(batch), len(self.dataset.entities)], dtype=torch.float32)
        targets = targets.cuda()
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        return np.array(batch), targets

    def evaluate(self, model, test_data):
        model.eval()
        hits = [[]*10]
        ranks = []
        test_data_idxs = self.get_data_idxs(test_data)
        hl_vocab_t = self.get_hl_t(test_data_idxs)
        for i in tqdm(range(0, len(test_data_idxs), self.batch_size)):
            data_batch = np.array(test_data_idxs[i: i + self.batch_size])
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            e1_idx = e1_idx.cuda()
            r_idx = r_idx.cuda()
            e2_idx = e2_idx.cuda()
            predictions = model.get_scores(e1_idx, r_idx)
            for j in range(data_batch.shape[0]):
                filt = hl_vocab_t[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
        hitat10 = np.mean(hits[9])
        hitat3 = np.mean(hits[2])
        hitat1 = np.mean(hits[0])
        meanrank = np.mean(ranks)
        mrr = np.mean(1. / np.array(ranks))
        print('Hits @10: {0}'.format(hitat10))
        print('Hits @3: {0}'.format(hitat3))
        print('Hits @1: {0}'.format(hitat1))
        print('Mean rank: {0}'.format(meanrank))
        print('Mean reciprocal rank: {0}'.format(mrr))
        return [mrr, meanrank, hitat10, hitat3, hitat1]

    def save_checkpoint(self, model, model_dir):
        self.write_vocab_files(model_dir)
        self.write_embedding_files(model, model_dir)

    def write_vocab_files(self, model_dir):
        with open(os.path.join(model_dir, 'idx_entities.dict'),'wt',encoding='utf-8') as out_ie, \
                open(os.path.join(model_dir, 'entities_idx.dict'),'wt',encoding='utf-8') as out_ei:
            for idx, entity in enumerate(self.dataset.entities):
                out_ie.write(str(idx)+'\t'+entity+'\n')
                out_ei.write(entity+'\t'+str(idx)+'\n')
        with open(os.path.join(model_dir, 'idx_relations.dict'), 'wt', encoding='utf-8') as out_ir, \
                open(os.path.join(model_dir, 'relations_idx.dict'), 'wt', encoding='utf-8') as out_ri:
            for idx, relation in enumerate(self.dataset.relations):
                out_ir.write(str(idx) + '\t' + relation + '\n')
                out_ri.write(relation + '\t' + str(idx) + '\n')

    @staticmethod
    def write_embedding_files(model, model_dir):
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
        np.save(model_dir + '/E.npy', model.E.weight.data.cpu().numpy())
        np.save(model_dir + '/R.npy', model.R.weight.data.cpu().numpy())
        np.save(model_dir + '/head_bn.npy', {'weight': model.head_bn.weight.data.cpu().numpy(),
                                             'bias': model.head_bn.bias.data.cpu().numpy(),
                                             'running_mean': model.head_bn.running_mean.data.cpu().numpy(),
                                             'running_var': model.head_bn.running_var.data.cpu().numpy()})
        np.save(model_dir + '/score_bn.npy', {'weight': model.score_bn.weight.data.cpu().numpy(),
                                              'bias': model.score_bn.bias.data.cpu().numpy(),
                                              'running_mean': model.score_bn.running_mean.data.cpu().numpy(),
                                              'running_var': model.score_bn.running_var.data.cpu().numpy()})

    def train_and_eval(self):
        best_eval = [0] * 5
        train_data_idxs = self.get_data_idxs(self.dataset.train_data)
        print(f'dataset: {self.dataset_name}, entities: {len(self.dataset.entities)}, relations: \
        {len(self.dataset.relations)}, training data count: {len(self.dataset.train_data)}.')
        model = ComplEx_KGE(self.dataset, self.ent_vec_dim, do_batch_norm=self.do_batch_norm,
                            input_dropout=self.input_dropout, hidden_dropout1=self.hidden_dropout1,
                            hidden_dropout2=self.hidden_dropout2)
        if self.load_from:
            model.load_state_dict(torch.load(self.load_from))
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=self.decay_rate) if 1 > self.decay_rate > 0 else None
        er_vocab = self.get_hl_t(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        print("starting training...")
        start_train = time.time()
        for epo in range(self.num_epochs):
            epoch_idx = epo + 1
            start_epoch = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                optimizer.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0]).cuda()
                r_idx = torch.tensor(data_batch[:, 1]).cuda()
                loss = model(e1_idx, r_idx, targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print('epoch:', epoch_idx, 'epoch time:', time.time() - start_epoch, 'loss:', np.mean(losses))
            if epoch_idx % self.test_interval == 0:
                model.eval()
                with torch.no_grad():
                    start_test = time.time()
                    print('validation results:')
                    valid_res = self.evaluate(model, self.dataset.valid_data)  # mrr, meanrank, hitat10, hitat3, hitat1
                    print('test results:')
                    test_res = self.evaluate(model, self.dataset.test_data)  # mrr, meanrank, hitat10, hitat3, hitat1
                    eval_res = (np.add(test_res, valid_res))/2
                    if eval_res[0] >= best_eval[0]:
                        best_eval = eval_res
                        print(f'evaluation MRR increased, saving checkpoint to {self.best_model_save_dir}')
                        self.save_checkpoint(model, model_dir=self.best_model_save_dir)
                        print('best model saved!')
                        print('overall best evaluation:', best_eval)
                    print(f'test time cost: [{time.time() - start_test}]')
            if scheduler:
                scheduler.step()
        print(f'training over, saving checkpoint to {self.final_model_save_dir}')
        self.save_checkpoint(model, model_dir=self.final_model_save_dir)
        print('final model saved!')
        print(f'total time cost: [{time.time() - start_train}]')


if __name__ == '__main__':
    set_fixed_seed(seed=199839)
    experiment = Experiment(num_epochs=500, test_interval=10, batch_size=128, learning_rate=0.0005, ent_vec_dim=200,
                            rel_vec_dim=200, input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                            label_smoothing=0.1, do_batch_norm=True, data_dir='./knowledge_graphs',
                            model_dir='./kg_embeddings', dataset_name='FB15k-237', load_from='', decay_rate=1.0)
    experiment.train_and_eval()
