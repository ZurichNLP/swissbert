from collections import Counter
from typing import Tuple, List, Union

import torch
from networkx.algorithms.bipartite import from_biadjacency_matrix
from scipy.sparse import csr_matrix
from tqdm import tqdm

from evaluation.romansh_alignment.utils import LayerAggregation, subword_to_word_map, AlignmentLevel, WordAlignment
from evaluation.romansh_alignment.word_aligners import WordAligner, AlignerOutput


class SimalignMethod(str):
    ARGMAX = "argmax"
    ITERMAX = "itermax"
    MATCH = "match"


class SimalignAligner(WordAligner):
    """
    https://www.aclweb.org/anthology/2020.findings-emnlp.147.pdf – Argmax only
    Average the subword embeddings to obtain word embeddings
    """

    def __init__(self,
                 tokenizer,
                 model,
                 tokenizer_args: dict = None,
                 model_args: dict = None,
                 layer: int = -1,
                 method: Union[str, SimalignMethod] = SimalignMethod.ARGMAX,
                 aggregation: Union[str, LayerAggregation] = LayerAggregation.SINGLE,
                 ):
        if method == SimalignMethod.ITERMAX:
            raise NotImplementedError
        self.method = method
        self.aggregation = aggregation

        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args or {
            "return_tensors": "pt",
        }
        self.model = model
        self.model.eval()
        self.model_args = model_args or {
            "output_hidden_states": True,
            "return_dict": True,
        }
        self.layer = layer

    def align(self,
              src_sentences: List[str],
              tgt_sentences: List[str],
              src_lang_id: int = None,
              tgt_lang_id: int = None,
              ) -> AlignerOutput:
        assert len(src_sentences) == len(tgt_sentences)

        alignments = []
        all_src_words = []
        all_tgt_words = []
        for i in tqdm(list(range(len(src_sentences)))):
            src_sentence = src_sentences[i]
            all_src_words.append(src_sentence.split())
            src_embeddings = self._encode_sentence(src_sentence, src_lang_id)

            tgt_sentence = tgt_sentences[i]
            all_tgt_words.append(tgt_sentence.split())
            tgt_embeddings = self._encode_sentence(tgt_sentence, tgt_lang_id)
            
            similarity_matrix = self._get_similarity_matrix(src_embeddings, tgt_embeddings)

            if self.method == SimalignMethod.ARGMAX:
                forward, backward = self._get_alignment_matrix(similarity_matrix)
                aligns = forward & backward
            elif self.method == SimalignMethod.MATCH:
                aligns = self._get_max_weight_match(similarity_matrix)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            word_alignment = WordAlignment.from_labels(aligns[0].cpu().detach().numpy())
            alignments.append(word_alignment)

        return AlignerOutput(
            alignments=alignments,
            src_tokens=all_src_words,
            tgt_tokens=all_tgt_words,
            level=AlignmentLevel.WORD,
        )

    @torch.no_grad()
    def _encode_sentence(self, sentence: str, lang_id: int = None) -> torch.Tensor:
        """
        Encode a sentence and return the word embeddings, averaged across the subwords of a word
        """
        words = sentence.split()
        encoding = self.tokenizer(sentence, **self.tokenizer_args).to(self.model.device)
        to_word_map = subword_to_word_map(sentence, encoding)
        model_args = self.model_args.copy()
        if lang_id is not None:
            model_args["lang_ids"] = torch.tensor([lang_id], device=self.model.device)
        output = self.model(**encoding, **model_args)

        if self.aggregation == LayerAggregation.SINGLE:
            hidden_states = output.hidden_states[self.layer]
        elif self.aggregation == LayerAggregation.AVERAGE:
            hidden_states = torch.stack(output.hidden_states, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        word_embeddings = torch.zeros((1, len(words), hidden_states.size(-1)), device=self.model.device)
        for token_id, word_id in enumerate(to_word_map):
            if word_id is None:
                continue
            word_embeddings[0, word_id] += hidden_states[0, token_id]
        subwords_counter = Counter(to_word_map)
        for word_id, count in subwords_counter.items():
            if word_id is None:
                continue
            word_embeddings[0, word_id] /= count
        return word_embeddings

    @staticmethod
    @torch.no_grad()
    def _get_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Initial source: https://stackoverflow.com/a/58144658/3902795
        Added a batch dimension
        """
        eps = 1e-8
        # Initial dim: batch x seq_len x embedding_size
        a_n, b_n = a.norm(dim=2)[..., None], b.norm(dim=2)[..., None]  # Same dim
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))  # Same dim
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        pairwise_cosine_similarity = torch.matmul(a_norm, b_norm.transpose(1, 2))  # batch x seq_len_1 x seq_len_2
        return (pairwise_cosine_similarity + 1.0) / 2.0  # Same dim

    @staticmethod
    @torch.no_grad()
    def _get_alignment_matrix(similarity_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_1, seq_len_2 = similarity_matrix.shape
        forward_base = torch.eye(seq_len_2, dtype=torch.bool, device=similarity_matrix.device)
        backward_base = torch.eye(seq_len_1, dtype=torch.bool, device=similarity_matrix.device)
        forward = forward_base[similarity_matrix.argmax(dim=2)]  # batch x seq_len_1 x seq_len_2
        backward = backward_base[similarity_matrix.argmax(dim=1)]  # batch x seq_len_2 x seq_len_1
        return forward, backward.transpose(1, 2)

    @staticmethod
    @torch.no_grad()
    def _get_max_weight_match(similarity_matrix: torch.Tensor) -> torch.Tensor:
        try:
            import networkx as nx
        except ImportError:
            raise ValueError("networkx must be installed to use match algorithm.")

        similarity_matrix = similarity_matrix.squeeze(0)

        def permute(edge):
            if edge[0] < similarity_matrix.shape[0]:
                return edge[0], edge[1] - similarity_matrix.shape[0]
            else:
                return edge[1], edge[0] - similarity_matrix.shape[0]

        G = from_biadjacency_matrix(csr_matrix(similarity_matrix.cpu().numpy()))
        matching = nx.max_weight_matching(G, maxcardinality=True)
        matching = [permute(x) for x in matching]
        matching = sorted(matching, key=lambda x: x[0])
        res_matrix = torch.zeros_like(similarity_matrix, device=similarity_matrix.device)
        for edge in matching:
            res_matrix[edge[0], edge[1]] = 1
        return res_matrix.unsqueeze(0)

    def __str__(self):
        return f"SimalignAligner({self.model.name_or_path.replace('/', '_')})"
