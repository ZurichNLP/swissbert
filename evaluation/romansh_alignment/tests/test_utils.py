from unittest import TestCase

import transformers
from transformers import AutoModel, AutoTokenizer

from evaluation.romansh_alignment import utils
from evaluation.romansh_alignment.dermit import DermitCorpus
from evaluation.romansh_alignment.encoders.hf import HuggingfaceEncoder
from evaluation.romansh_alignment.utils import WordAlignment, AlignmentLevel


class WordAlignmentTestCase(TestCase):

    def setUp(self) -> None:
        self.src_token_to_word = [0, 1, 2, 3, 4, 5]
        self.tgt_token_to_word = [0, 1, 2, 3, 3, 4, 5]

    def test_init(self):
        self.assertSetEqual(
            WordAlignment([(0, 0), (1, 1)]),
            WordAlignment.fromstring("0-0 1-1"),
        )
        self.assertEqual(
            str(WordAlignment([(0, 0), (1, 1)])),
            str(WordAlignment.fromstring("0-0 1-1")),
        )

    def test_level(self):
        self.assertEqual(AlignmentLevel.WORD, WordAlignment([]).level)


class AlignmentUtilsTestCase(TestCase):

    def test_subword_to_word_map(self):
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        sentence = "Coronavirus : Pussaivladad da far l' annunzia per la vaccinaziun d' uffants"
        encoding = tokenizer(sentence, return_tensors="pt")
        subword_to_word_map = utils.subword_to_word_map(sentence, encoding)
        self.assertEqual(len(set([i for i in subword_to_word_map if i is not None])), len(sentence.split()))
        self.assertEqual(len(subword_to_word_map), encoding.input_ids.shape[1])
        self.assertEqual(set([i for i in subword_to_word_map if i is not None]), set(list(range(len(sentence.split())))))

    def test_subword_to_word_map__custom_vocab(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained("ZurichNLP/swissbert")
        corpus = DermitCorpus()
        for sentence in corpus.de_sentences + corpus.rm_sentences:
            encoding = tokenizer(sentence, return_tensors="pt")
            subword_to_word_map = utils.subword_to_word_map(sentence, encoding)
            self.assertEqual(len(set([i for i in subword_to_word_map if i is not None])), len(sentence.split()))
            self.assertEqual(len(subword_to_word_map), encoding.input_ids.shape[1])
            self.assertEqual(set([i for i in subword_to_word_map if i is not None]), set(list(range(len(sentence.split())))))


class BERTScoreTestCase(TestCase):

    def setUp(self) -> None:
        model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        self.encoder = HuggingfaceEncoder(
            model=model,
            tokenizer=tokenizer,
        )

    def test_bert_score(self):
        text1 = "This is a test."
        text2 = "That is an attempt."
        text3 = "And here comes a completely unrelated sentence!"
        embeddings1 = self.encoder.embed_tokens(text1)
        embeddings2 = self.encoder.embed_tokens(text2)
        embeddings3 = self.encoder.embed_tokens(text3)
        score12 = utils.bert_score(embeddings1, [embeddings2])[0].item()
        score13 = utils.bert_score(embeddings1, [embeddings3])[0].item()
        score23 = utils.bert_score(embeddings2, [embeddings3])[0].item()
        self.assertGreater(score12, score13)
        self.assertGreater(score12, score23)

    def test_bert_score_batched(self):
        text1 = "This is a test."
        text2 = "That is an attempt."
        text3 = "And here comes a completely unrelated sentence!"
        embeddings1 = self.encoder.embed_tokens(text1)
        embeddings2 = self.encoder.embed_tokens(text2)
        embeddings3 = self.encoder.embed_tokens(text3)
        scores1 = utils.bert_score(embeddings1, [embeddings2, embeddings3])
        scores2 = utils.bert_score(embeddings2, [embeddings1, embeddings3])
        scores3 = utils.bert_score(embeddings3, [embeddings1, embeddings2])
        self.assertEqual(scores1[0].item(), scores2[0].item())
        self.assertEqual(scores1[1].item(), scores3[0].item())
        self.assertEqual(scores2[1].item(), scores3[1].item())
