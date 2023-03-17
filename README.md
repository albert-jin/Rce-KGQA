# Rce-KGQA
A novel pipeline framework for multi-hop complex KGQA task.

This framework mainly contains two modules, **answering_filtering_module** and **relational_chain_reasoning_module**

And this two module should be trained independently, at reference step, question and KG load into **answering_filtering_module** ad inputs, then get the top-K candidates
,and retrieval these candidates` relational chain in KG, and let **relational_chain_reasoning_module** provide the final answer to USERS.
> overall pipeline architecture 
[See model](https://github.com/albert-jin/Rce-KGQA/blob/main/intros/all_architecture.pdf)

>answering_filtering_module
[See Module1](https://github.com/albert-jin/Rce-KGQA/blob/main/intros/answer_filtering.pdf)

>relational_chain_reasoning_module
[See Module2](https://github.com/albert-jin/Rce-KGQA/blob/main/intros/relational_chain_reasoning.pdf)

Statistical Performance Comparsion:

### Experimental results on three subsets of MetaQA. The first group of results was taken from papers on recent methods. The values are
reported using hits@1.

| Model | 1-hop MetaQA | 2-hop MetaQA | 3-hop MetaQA ||

| :-----| ----: | :----: ||

| EmbedKGQA | 97.5 | 98.8 | 94.8 ||

| SRN | 97.0 | 95.1 | 75.2 ||

| KVMem | 96.2 |  82.7 |  48.9 ||

| GraftNet | 97.0 |  94.8 |  77.7 ||

| PullNet | 97.0 | 99.9 | 91.4 ||

| Our Model | 98.3 | 99.7 | 97.9 ||

### Experimental results on Answer Reasoning on WebQuestionsSP-tiny.

 Experiment results compared with SOTA methods on WebQuestionsSP-tiny test set. All QA pairs in WebQuestionsSP-tiny are 2-hop
relational questions.

| Model | WebQuestionsSP-tiny hit@1 ||

| EmbedKGQA | 66.6 ||

| SRN | - ||

| KVMem | 46.7 ||

| GraftNet | 66.4 ||

| PullNet | 68.1 ||

| Our Model | 70.4 ||

Hope you enjoy it !!!  Arxiv link: https://arxiv.org/abs/2110.12679

If this work helps you, please cite it. thanks!

```
@Article{jwq2022rcekgqa,
author={Jin, Weiqiang
and Zhao, Biao
and Yu, Hang
and Tao, Xi
and Yin, Ruiping
and Liu, Guizhong},
title={Improving embedded knowledge graph multi-hop question answering by introducing relational chain reasoning},
journal={Data Mining and Knowledge Discovery},
year={2022},
month={Nov},
day={11},
issn={1573-756X},
doi={10.1007/s10618-022-00891-8},
url={https://doi.org/10.1007/s10618-022-00891-8}
}
```
