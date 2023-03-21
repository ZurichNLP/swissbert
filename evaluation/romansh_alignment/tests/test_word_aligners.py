from unittest import TestCase

from transformers import AutoTokenizer, AutoModel

from evaluation.romansh_alignment.dermit import DermitCorpus
from evaluation.romansh_alignment.word_aligners.simalign_aligner import SimalignAligner


class SimalignAlignerTestCase(TestCase):

    def setUp(self) -> None:
        self.model_name = "google/bert_uncased_L-2_H-128_A-2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.corpus = DermitCorpus()
        self.aligner = SimalignAligner(
            tokenizer=self.tokenizer,
            model=self.model,
        )

    def test_align_argmax(self):
        output = self.aligner.align(
            src_sentences=self.corpus.get_de_sentences()[:3],
            tgt_sentences=self.corpus.get_rm_sentences()[:3],
        )
        print(output)
        self.assertEqual(3, len(output.alignments))
        self.assertEqual(3, len(output.src_tokens))
        self.assertEqual(3, len(output.tgt_tokens))

    def test_align_match(self):
        self.aligner.method = "match"
        output = self.aligner.align(
            src_sentences=self.corpus.get_de_sentences()[:3],
            tgt_sentences=self.corpus.get_rm_sentences()[:3],
        )
        print(output)
        self.assertEqual(3, len(output.alignments))
        self.assertEqual(3, len(output.src_tokens))
        self.assertEqual(3, len(output.tgt_tokens))
