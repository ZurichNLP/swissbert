from pathlib import Path
from typing import List

from evaluation.romansh_alignment.utils import WordAlignment


class DermitCorpus:

    def __init__(self, repo_dir: Path = None):
        if repo_dir is None:
            repo_dir = Path(__file__).parent / "DERMIT-Corpus"
        assert repo_dir.exists()
        self.repo_dir = repo_dir
        de_sentences_path = self.repo_dir / "gold_standard" / "alignments_src_text.txt"
        rm_sentences_path = self.repo_dir / "gold_standard" / "alignments_trg_text.txt"
        alignments_path = self.repo_dir / "gold_standard" / "alignments_pharaoh.txt"
        self.de_sentences: List[str] = []
        self.rm_sentences: List[str] = []
        alignment_strs: List[str] = []
        with open(de_sentences_path) as f_de, open(rm_sentences_path) as f_rm, open(alignments_path) as f_alignments:
            for de_sentence, rm_sentence, alignment in zip(f_de, f_rm, f_alignments):
                de_idx, de_sentence = de_sentence.split("\t")
                rm_idx, rm_sentence = rm_sentence.split("\t")
                alignment_idx, alignment = alignment.split("\t")
                assert de_idx == rm_idx == alignment_idx
                self.de_sentences.append(self._clean_sentence(de_sentence))
                self.rm_sentences.append(self._clean_sentence(rm_sentence))
                alignment_strs.append(alignment.strip())

        # Parse word alignments
        self.word_alignments = [WordAlignment.fromstring(s) for s in alignment_strs]

        # De-duplicate the corpus
        duplicate_indices = set()
        for i in range(len(self.de_sentences)):
            for j in range(i + 1, len(self.de_sentences)):
                if self.de_sentences[i] == self.de_sentences[j]:
                    duplicate_indices.add(j)
        for i in range(len(self.rm_sentences)):
            for j in range(i + 1, len(self.rm_sentences)):
                if self.rm_sentences[i] == self.rm_sentences[j]:
                    duplicate_indices.add(j)
        print(f"Removing {len(duplicate_indices)} duplicate sentences:")
        for i in sorted(duplicate_indices, reverse=True):
            print(f"  {self.de_sentences[i]}")
            print(f"  {self.rm_sentences[i]}")
            print()
            del self.de_sentences[i]
            del self.rm_sentences[i]
            del self.word_alignments[i]

    @staticmethod
    def _clean_sentence(sentence: str) -> str:
        sentence = sentence.replace("   ", " ")
        sentence = sentence.replace("  ", " ")
        diaresis = "\u0308"
        sentence = sentence.replace(f"a{diaresis}", "ä")
        sentence = sentence.replace(f"o{diaresis}", "ö")
        sentence = sentence.replace(f"u{diaresis}", "ü")
        sentence = sentence.replace(f"A{diaresis}", "Ä")
        sentence = sentence.replace(f"O{diaresis}", "Ö")
        sentence = sentence.replace(f"U{diaresis}", "Ü")
        return sentence.strip()

    def __len__(self):
        return len(self.de_sentences)

    def get_de_sentences(self) -> List[str]:
        return self.de_sentences

    def get_rm_sentences(self) -> List[str]:
        return self.rm_sentences

    def get_word_alignments(self) -> List[WordAlignment]:
        return self.word_alignments
