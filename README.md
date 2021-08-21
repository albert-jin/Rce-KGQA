# Rce-KGQA
A novel pipeline framework for multi-hop complex KGQA task.

This framework mainly contains two modules, **answering_filtering_module** and **relational_chain_reasoning_module**

And this two module should be trained independently, at reference step, question and KG load into **answering_filtering_module** ad inputs, then get the top-K candidates
,and retrieval these candidates` relational chain in KG, and let **relational_chain_reasoning_module** provide the final answer to USERS.
> overall pipeline architecture
> <img src="https://github.com/albert-jin/Rce-KGQA/raw/main/intros/all_architecture.pdf" width="50%">

>answering_filtering_module
<img src="https://github.com/albert-jin/Rce-KGQA/raw/main/intros/answer_filtering.pdf" width="50%">

>relational_chain_reasoning_module
<img src="https://github.com/albert-jin/Rce-KGQA/raw/main/intros/relational_chain_reasoning.pdf" width="50%">