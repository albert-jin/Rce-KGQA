# Rce-KGQA
A novel pipeline framework for multi-hop complex KGQA task.

This framework mainly contains two modules, **answering_filtering_module** and **relational_chain_reasoning_module**

And this two module should be trained independently, at reference step, question and KG load into **answering_filtering_module** ad inputs, then get the top-K candidates
,and retrieval these candidates` relational chain in KG, and let **relational_chain_reasoning_module** provide the final answer to USERS.
> overall pipeline architecture 
[See model](https://github.com/albert-jin/Rce-KGQA/main/intros/all_architecture.pdf)

>answering_filtering_module
[See Module1](https://github.com/albert-jin/Rce-KGQA/main/intros/answer_filtering.pdf)

>relational_chain_reasoning_module
[See Module2](https://github.com/albert-jin/Rce-KGQA/main/intros/relational_chain_reasoning.pdf)


Hope you enjoy it !!!