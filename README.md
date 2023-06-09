<a href="https://arxiv.org/abs/2303.13310" target="_blank"><img src="http://img.shields.io/badge/arXiv-2303.13310-orange.svg?style=flat" alt="http://img.shields.io/badge/arXiv-2303.13310-orange.svg?style=flat"></a>

<img src="swissbert-diagram.png" alt="SwissBERT is a transformer encoder with language adapters in each layer. There is an adapter for each national language of Switzerland. The other parameters in the model are shared among the four languages." width="450">

SwissBERT is a masked language model for processing Switzerland-related text. It has been trained on more than 21 million Swiss news articles retrieved from [Swissdox@LiRI](https://t.uzh.ch/1hI).

Based on [X-MOD](https://github.com/facebookresearch/fairseq/tree/main/examples/xmod), which has been pre-trained with language adapters in 81 languages.
SwissBERT contains adapters for the national languages of Switzerland – German, French, Italian, and Romansh Grischun.
In addition, it uses a Switzerland-specific subword vocabulary.

The easiest way to use SwissBERT is via the [transformers](https://github.com/huggingface/transformers) library and the Hugging Face model hub: https://huggingface.co/ZurichNLP/swissbert

## License
- This code repository: MIT license
- Model: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

## Pre-training code
See [pretraining](pretraining)

## Evaluation code

### SwissNER
See [evaluation/swissner/notebook.ipynb](evaluation/swissner/notebook.ipynb)

### HIPE-2022
See [evaluation/hipe2022/notebook.ipynb](evaluation/hipe2022/notebook.ipynb)

### X-Stance
See [evaluation/xstance/notebook.ipynb](evaluation/xstance/notebook.ipynb)

### German–Romansh alignment
See [evaluation/romansh_alignment/notebook.ipynb](evaluation/romansh_alignment/notebook.ipynb)

## Citation
```bibtex
@article{vamvas-etal-2023-swissbert,
      title={Swiss{BERT}: The Multilingual Language Model for Switzerland}, 
      author={Jannis Vamvas and Johannes Gra\"en and Rico Sennrich},
      year={2023},
      eprint={2303.13310},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2303.13310}
}
```
