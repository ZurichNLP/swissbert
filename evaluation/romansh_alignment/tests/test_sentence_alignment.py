from unittest import TestCase

from evaluation.romansh_alignment.sentence_alignment import SentenceAlignmentBenchmark


class SentenceAlignmentBenchmarkTestCase(TestCase):

    def setUp(self):
        self.benchmark = SentenceAlignmentBenchmark(
            limit_n=10,
        )

    def test_hf_bert_score(self):
        from transformers import AutoModel
        model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        from ..encoders.hf import HuggingfaceEncoder
        encoder = HuggingfaceEncoder(
            model=model,
            tokenizer=tokenizer,
        )
        result = self.benchmark.evaluate_encoder_bert_score(encoder)
        print(result)
