from unittest import TestCase

from evaluation.romansh_alignment.dermit import DermitCorpus


class DermitCorpusTestCase(TestCase):

    def setUp(self) -> None:
        self.corpus = DermitCorpus()

    def test_len(self):
        self.assertEqual(597, len(self.corpus))

    def test_sentence(self):
        self.assertEqual("Coronavirus : Anmeldung zur Kinderimpfung möglich", self.corpus.de_sentences[0])

    def test_word_alignments(self):
        self.assertEqual("0-0 1-1 2-6 3-7 4-9 4-11 5-2", str(self.corpus.word_alignments[0]))
