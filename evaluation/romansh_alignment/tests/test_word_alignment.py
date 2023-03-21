from unittest import TestCase

from transformers import AutoTokenizer, AutoModel

from evaluation.romansh_alignment.word_aligners.simalign_aligner import SimalignAligner
from evaluation.romansh_alignment.word_alignment import WordAlignmentBenchmark


class WordAlignmentBenchmarkTestCase(TestCase):

    def setUp(self) -> None:
        self.benchmark = WordAlignmentBenchmark(
            limit_n=10,
        )

    def test_simalign(self):
        self.model_name = "google/bert_uncased_L-2_H-128_A-2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.aligner = SimalignAligner(
            tokenizer=self.tokenizer,
            model=self.model,
            aggregation="average",
        )
        result = self.benchmark.evaluate(self.aligner)
        print(result)
