# Transformers-for-Language-Modelling
This repo consists of different codebases of transformers and their applications in recent Large Language Models, such as GPT, GPT2, etc. These codebases are coded and trained from scratch and replicated from different papers and authors. 

**Why do you need this repo?** 🤔  <br>-
There is not much documented, not good enough, if it is, to follow through and understand each bit of code so that people can reproduce something of their own. This repo has (overstating because if not, it will be updated) all the mechanisms in the most advanced and sought-after papers and language models.

## What's in the repository so far 🚀
- **Transformers from Scratch** <br>
Replicated from `Attention is all you need` by Vaswani 2017, the pioneer of transformers. The configurations match the original paper; however, due to memory and GPU constraints, this version of the transformer is trained on *500 Epochs*. <br>
It follows the following configurations: <br>
`Block_size=8, Batch_size=32, emb_dim=256, head_size=32, num_heads=8, num_layers=2, vocab_size= vocab size of the tokenizer`. <br>
The architecture follows **Multi-Headed Self Attention**.



