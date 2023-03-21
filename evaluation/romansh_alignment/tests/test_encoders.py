from unittest import TestCase

from transformers import AutoModel, AutoTokenizer

from evaluation.romansh_alignment.encoders.hf import HuggingfaceEncoder


class HuggingfaceEncoderTestCase(TestCase):

    def setUp(self):
        model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        self.encoder = HuggingfaceEncoder(
            model=model,
            tokenizer=tokenizer,
            aggregation="average",
        )

    def test_embed_sentence(self):
        text = "The quick brown fox jumps over the lazy dog."
        embeddings = self.encoder.embed_sentence(text)
        self.assertEqual(embeddings.shape, (128,))
