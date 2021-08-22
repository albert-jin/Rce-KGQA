import torch
import networkx
import os
import logging
import time
from dataloader import DEV_MetaQADataLoader
from model import Relational_chain_reasoning_module
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# from answer_filtering_module.model import Answer_filtering_module

# ====dataloader prepare=====
KG_NAME = 'MetaQA'
assert KG_NAME in ['MetaQA', 'FB15k-237', 'WebquestionsSP-tiny']
HOPS = 1
assert HOPS in [1, 2, 3]
KG_HALF = True
assert type(KG_HALF) is bool
if KG_NAME == 'MetaQA':
    QA_PAIRS_PATH = f'../QA/MetaQA/{"half_" if KG_HALF else ""}qa_%s_{str(HOPS)}hop.txt'
else:
    QA_PAIRS_PATH = f'../QA/WebQuestionsSP/qa_%s_webqsp.txt'
qa_traindataset_path = QA_PAIRS_PATH % 'train'
qa_devdataset_path = QA_PAIRS_PATH % 'dev'
qa_testdataset_path = QA_PAIRS_PATH % 'test'
batch_size = 4
BEST_OR_FINAL = 'best'
assert BEST_OR_FINAL in ['best', 'final']
KG_EMBED_PATH = f'../knowledge_graph_embedding_module/kg_embeddings/{KG_NAME}{"_half" if KG_HALF else "_full"}/' \
                f'{BEST_OR_FINAL}_checkpoint/%s'
E_path = KG_EMBED_PATH % 'E.npy'
RELATION_EMBEDDINGS_PATH = KG_EMBED_PATH % 'R.npy'
RELATION_EMBEDDINGS = np.load(RELATION_EMBEDDINGS_PATH)
ENTITY_DICT_PATH = KG_EMBED_PATH % 'entities_idx.dict'
RELATION_DICT_PATH = KG_EMBED_PATH % 'relations_idx.dict'
ENTITY_DICT = dict()
with open(ENTITY_DICT_PATH, mode='rt', encoding='utf-8') as inp:
    for line in inp:
        split_infos = line.strip().split('\t')
        ENTITY_DICT[split_infos[0]] = split_infos[1]
RELATION_DICT = dict()
with open(RELATION_DICT_PATH, mode='rt', encoding='utf-8') as inp:
    for line in inp:
        split_infos = line.strip().split('\t')
        RELATION_DICT[split_infos[0]] = split_infos[1]
# ==result store==
TRAINING_RESULTS_DIR = os.path.join('.', '_'.join([KG_NAME, "half" if KG_HALF else "full", str(HOPS), 'hop',
                                                   time.asctime().replace(' ', '_').replace(':', '_')]))
if not os.path.isdir(TRAINING_RESULTS_DIR):
    os.mkdir(TRAINING_RESULTS_DIR)
best_model_path = os.path.join(TRAINING_RESULTS_DIR, 'best_rcrm_model.pt')
final_model_path = os.path.join(TRAINING_RESULTS_DIR, 'final_rcrm_model.pt')

# ==logger==
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)
file_logger = logging.FileHandler(os.path.join(TRAINING_RESULTS_DIR, 'training.log'), encoding='UTF-8')
file_logger.setLevel(logging.INFO)
file_logger.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_logger)

logger.info(
    f'Meta Infos ===> KG name:[{KG_NAME}], Gop:[{HOPS}], Is_half:[{KG_HALF}], Relation counts:[{len(RELATION_DICT)}], Entity '
    f'counts:[{len(ENTITY_DICT)}]')

# ====prepare KG====
KG_GRAPH_PATH = f'../knowledge_graph_embedding_module/knowledge_graphs/{KG_NAME}{"_half" if KG_HALF else "_full"}' \
                f'%s.txt'


def load_data(data_path, reverse):
    with open(data_path, 'r', encoding='utf-8') as inp_:
        data = [l.strip().split('\t') for l in inp_.readlines()]
        if reverse:
            data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        new_data = []
        for i in data:
            new_data.append([(ENTITY_DICT[i[0]], {'name': i[0]}), {'r_idx': RELATION_DICT[i[1]], 'r': i[1]},
                             (ENTITY_DICT[i[2]], {'name': i[2]})])
    return new_data


all_KG_triples = load_data(KG_GRAPH_PATH % 'train', True) + load_data(KG_GRAPH_PATH % 'valid', True) + load_data(
    KG_GRAPH_PATH % 'test', True)

KG = networkx.DiGraph()
"""
        Use (node, attrdict) tuples to update attributes for specific nodes.

        >>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])
        >>> G.nodes[1]["size"]
        11
"""
KG.add_nodes_from(map(lambda x: (x[0], x[2]), all_KG_triples))  # store all nodes and their 'name' attributes.
KG.add_edges_from(map(lambda x: (x[0][0], x[2][0], x[1]), all_KG_triples))  # store all edges and edges` type [r_idx, r]
logger.info(KG.number_of_edges(), KG.number_of_nodes())

# ==load answer_filtering_module ==
AFM_STORE_PATH = f'../answer_filtering_module/{"_".join([KG_NAME, "half" if KG_HALF else "full", str(HOPS), "hop", "2021_07_12_12_04"])}/best_afm_model.pt'
afm_model = torch.load(AFM_STORE_PATH)
afm_model.to(device=torch.device('cuda'))
afm_model.eval()
word_idx = dict()  # 用于afm模型上的语言tokenizer
with open(os.path.join(AFM_STORE_PATH, 'word_idx.txt'), mode='rt', encoding='utf-8') as inp:
    for line in inp:
        line = line.strip().split('\t')
        word_idx[line[0]] = int(line[1])

# ==load relational_chain_reasoning_module ==
load_from = ''
rcrm_model = Relational_chain_reasoning_module(relation_dim=RELATION_EMBEDDINGS.shape[-1], dim_l1=768,
                                               dim_l2=RELATION_EMBEDDINGS.shape[-1] * 2,
                                               lstm_hidden_dim=RELATION_EMBEDDINGS.shape[-1],
                                               relation_embeddings=RELATION_EMBEDDINGS)
if load_from:
    rcrm_model.load_state_dict(torch.load(load_from))
rcrm_model.to(device=torch.device('cuda'))
# ====prepare dataset====

afm_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=ENTITY_DICT_PATH,
                                      relation_dict_path=RELATION_DICT_PATH, qa_dataset_path=qa_traindataset_path)
afm_dev_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=ENTITY_DICT_PATH,
                                          relation_dict_path=RELATION_DICT_PATH, qa_dataset_path=qa_devdataset_path)
afm_test_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=ENTITY_DICT_PATH,
                                           relation_dict_path=RELATION_DICT_PATH, qa_dataset_path=qa_testdataset_path)
afm_process = tqdm(afm_dataloader, total=len(afm_dataloader), unit=' batches')
afm_dev_process = tqdm(afm_dev_dataloader, total=len(afm_dev_dataloader), unit=' batches')
afm_test_process = tqdm(afm_test_dataloader, total=len(afm_test_dataloader), unit=' batches')
# ====hyper-parameters prepare====
N_EPOCHS = 400
PATIENCE = 5
LR = 0.0001
adam_optimizer = torch.optim.Adam(rcrm_model.parameters(), lr=LR)
LR_DECAY = 0.95
TEST_INTERVAL = 5
USE_TOP_K = SUB_BATCH_SIZE = 32

# ====training step=====
for epoch_idx in range(N_EPOCHS):
    epoch_idx += 1
    avg_epoch_loss = 0
    afm_process.set_description('{}/{}'.format(epoch_idx, N_EPOCHS))
    for batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len, text_qs in afm_process:
        batch_ranked_topK_entity = afm_model.get_ranked_top_k(batch_questions_index, batch_questions_length,
                                                              batch_head_entity, max_sent_len, K=USE_TOP_K).tolist()
        """
        batch_ranked_topK_entity: 获得了batch_size 个question 在afm 模型的候选输出实体下标, shape: (batch_size, USE_TOP_K)
        batch_answers: 实际的正确答案集合, shape: (batch_size, random=>取决于答案数量)
        batch_head_entity: 每个question对应的topic entity的下标 shape: (batch_size, )
        接下来的step: 对每个question 的topK的候选实体,在KG中检索其关系链 relational chain, 对topK 中,属于答案的标为正样本,不属于答案的标记为负样本
        test_qs : batch_size 个问句的文本
        送给 rcrm_model ===> question_text, relational_chain_idxs, relation_chain_lengths, max_chain_len, label
        """
        rcrm_train_batch = []  # 收集给 rcrm_model 训练的数据
        for idx, head_entity in enumerate(batch_head_entity.tolist()):
            answers = batch_answers[idx]
            candids = batch_ranked_topK_entity[idx]
            text_q = text_qs[idx]
            for candid in candids:
                # 查询该topK个候选实体<=>topic实体之间的关系链
                e_path = networkx.shortest_path(KG, source=head_entity, target=candid)
                if len(e_path) < 2:
                    continue  # 如果topic 和 candidate 没有路径,则抛弃
                else:
                    relation_chain = []
                    for i in range(len(e_path) - 1):
                        # relation_chain.append((KG.edges[e_path[i], e_path[i+1]]['r_idx'], KG.edges[e_path[i], e_path[i+1]]['r']))
                        relation_chain.append(KG.edges[e_path[i], e_path[i + 1]]['r_idx'])
                if (candid not in answers) or (candid == head_entity):
                    rcrm_train_batch.append([text_q, relation_chain, 0])
                else:  # 只有在候选在answer中且不是topic才标记为正样本
                    rcrm_train_batch.append([text_q, relation_chain, 1])
        all_batch_count =len(rcrm_train_batch)
        sorted_batch = list(sorted(rcrm_train_batch, key=lambda x: len(x[1]), reverse=True))
        max_chain_len = len(sorted_batch[0][1])
        final_qs = []
        final_rc = []
        final_rcl = []
        final_label = []
        for text_q_, relation_chain_, label in sorted_batch:
            final_qs.append(text_q_)
            if len(relation_chain_) > 8:
                final_rc.append(relation_chain_[:8])
                final_rcl.append(8)
            else:
                final_rc.append(relation_chain_ + [0] * (8 - len(relation_chain_)))
                final_rcl.append(len(relation_chain_))
            final_label.append(label)
        final_rc = torch.tensor(final_rc, device=torch.device('cuda'))
        final_rcl = torch.tensor(final_rcl, device=torch.device('cuda'))
        final_label = torch.tensor(final_label, device=torch.device('cuda'))
        rcrm_model.zero_grad()
        loss = rcrm_model(question_text=final_qs, relational_chain_idxs=final_rc, relation_chain_lengths=final_rcl,
                          max_chain_len=max_chain_len, label=final_label, is_test=False)
        loss.backward()
        adam_optimizer.step()
        avg_iter_loss = loss.item() / all_batch_count
        avg_epoch_loss += avg_iter_loss
        afm_process.set_postfix(
            OrderedDict(Epoch=epoch_idx, Batch=all_batch_count, Batch_Loss=loss.item(), avg_Loss=avg_iter_loss))
        afm_process.update()
    logger.info(f'{epoch_idx}-th epoch: average_loss: {avg_epoch_loss / len(afm_dataloader) * batch_size}')

