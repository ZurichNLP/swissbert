from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from evaluation.romansh_alignment import utils
from evaluation.romansh_alignment.dermit import DermitCorpus
from evaluation.romansh_alignment.encoders import SentenceEncoder


@dataclass
class SentenceAlignmentResult:
    num_sentences: int
    accuracy: float


class SentenceAlignmentBenchmark:

    def __init__(self, limit_n: int = None):
        self.corpus = DermitCorpus()
        self.de_sentences = self.corpus.get_de_sentences()
        self.rm_sentences = self.corpus.get_rm_sentences()
        assert len(self.de_sentences) == len(self.rm_sentences)
        if limit_n is not None:
            self.de_sentences = self.de_sentences[:limit_n]
            self.rm_sentences = self.rm_sentences[:limit_n]

    def evaluate_encoder_bert_score(self,
                                    encoder: SentenceEncoder,
                                    src_lang_id: int = None,
                                    tgt_lang_id: int = None,
                                    top_k: int = 1,
                                    batch_size: int = 8,
                                    device=None,
                                    ) -> SentenceAlignmentResult:
        print(f"Encoding tokens with {encoder}...")
        de_embeddings = []
        for sentence in tqdm(self.de_sentences):
            de_embeddings.append(encoder.embed_tokens(sentence, lang_id=src_lang_id))
        rm_embeddings = []
        for sentence in tqdm(self.rm_sentences):
            rm_embeddings.append(encoder.embed_tokens(sentence, lang_id=tgt_lang_id))
        print("Calculating accuracy...")
        accuracy = self._calculate_bert_score_accuracy(de_embeddings, rm_embeddings,
                                                             top_k=top_k, device=(device or encoder.model.device),
                                                             batch_size=batch_size)
        print(f"Accuracy DE -> RM: {accuracy}")
        result = SentenceAlignmentResult(
            num_sentences=len(self.de_sentences),
            accuracy=accuracy,
        )
        return result

    def _calculate_bert_score_accuracy(self, all_query_embeddings, document_embeddings, top_k=1, batch_size=8, device="cpu") -> float:
        correct = 0
        for i, query_embeddings in enumerate(tqdm(all_query_embeddings)):
            scores = []
            # Batch document_embeddings to reduce memory usage
            for j in range(0, len(document_embeddings), batch_size):
                scores.extend(utils.bert_score(query_embeddings, document_embeddings[j:j+batch_size], device=device))
            scores = np.array(scores)
            correct += i in scores.argsort()[-top_k:]
        return correct / len(all_query_embeddings)
