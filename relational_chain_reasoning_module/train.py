import torch
import networkx
import os
import logging
import time
from dataloader import MetaQADataLoader, DEV_MetaQADataLoader
from model import Relational_chain_reasoning_module
import numpy as np
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
batch_size = 64
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

logger.info(f'Meta Infos ===> KG name:[{KG_NAME}], Gop:[{HOPS}], Is_half:[{KG_HALF}], Relation counts:[{len(RELATION_DICT)}], Entity '
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
            new_data.append([(ENTITY_DICT[i[0]], {'name': i[0]}), {'r_idx': RELATION_DICT[i[1]], 'r': i[1]}, (ENTITY_DICT[i[2]], {'name': i[2]})])
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
AFM_STORE_PATH = f'../answer_filtering_module/{"_".join([KG_NAME, "half" if KG_HALF else "full", str(HOPS), "hop","2021_07_12_12_04"])}/best_afm_model.pt'
afm_model = torch.load(AFM_STORE_PATH)
afm_model.to(device=torch.device('cuda'))
afm_model.eval()

# ==load relational_chain_reasoning_module ==
load_from = ''
rcrm_model = Relational_chain_reasoning_module(relation_dim=RELATION_EMBEDDINGS.shape[-1], dim_l1=768, dim_l2=RELATION_EMBEDDINGS.shape[-1]*2,
                                               lstm_hidden_dim=RELATION_EMBEDDINGS.shape[-1], relation_embeddings=RELATION_EMBEDDINGS)
if load_from:
    rcrm_model.load_state_dict(torch.load(load_from))
rcrm_model.to(device=torch.device('cuda'))
# ====prepare dataset====
qa_dataloader = MetaQADataLoader(entity_embed_path=E_path, entity_dict_path=ENTITY_DICT_PATH, relation_embed_path=RELATION_EMBEDDINGS_PATH
                                 , relation_dict_path=RELATION_DICT_PATH, qa_dataset_path=qa_traindataset_path,
                                 batch_size=batch_size)
word_idx = qa_dataloader.dataset.word_idx
qa_dev_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=ENTITY_DICT_PATH,
                                         relation_dict_path=RELATION_DICT_PATH, qa_dataset_path=qa_devdataset_path)
qa_test_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=ENTITY_DICT_PATH,
                                          relation_dict_path=RELATION_DICT_PATH, qa_dataset_path=qa_testdataset_path)

# ====hyper-parameters prepare====
N_EPOCHS = 400
PATIENCE = 5
LR = 0.0001
adam_optimizer = torch.optim.Adam(rcrm_model.parameters(), lr=LR)
