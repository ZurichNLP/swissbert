import logging
from dataclasses import dataclass

from evaluation.romansh_alignment.dermit import DermitCorpus
from evaluation.romansh_alignment.utils import AlignmentLevel, alignment_error_rate, alignment_f1_score
from evaluation.romansh_alignment.word_aligners import WordAligner


@dataclass
class WordAlignmentResult:
    aer: float
    precision: float
    recall: float
    f1: float


class WordAlignmentBenchmark:

    def __init__(self, limit_n: int = None):
        self.corpus = DermitCorpus()
        self.de_sentences = self.corpus.get_de_sentences()
        self.rm_sentences = self.corpus.get_rm_sentences()
        self.gold_alignments = self.corpus.get_word_alignments()
        assert len(self.de_sentences) == len(self.rm_sentences) == len(self.gold_alignments)
        if limit_n is not None:
            self.de_sentences = self.de_sentences[:limit_n]
            self.rm_sentences = self.rm_sentences[:limit_n]
            self.gold_alignments = self.gold_alignments[:limit_n]

    def evaluate(self, aligner: WordAligner, src_lang_id: int = None, tgt_lang_id: int = None) -> WordAlignmentResult:
        logging.info(f"Performing word alignment with {aligner}...")
        predictions = aligner.align(self.de_sentences, self.rm_sentences, src_lang_id, tgt_lang_id)
        assert predictions
        assert predictions.alignments
        assert predictions.level == AlignmentLevel.WORD
        aer = alignment_error_rate(self.gold_alignments, predictions.alignments)
        precision, recall, f1 = alignment_f1_score(self.gold_alignments, predictions.alignments)
        return WordAlignmentResult(
            aer=aer,
            precision=precision,
            recall=recall,
            f1=f1,
        )
