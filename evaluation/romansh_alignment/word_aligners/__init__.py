from dataclasses import dataclass
from typing import List

from nltk import Alignment
from transformers.utils import ModelOutput

from evaluation.romansh_alignment.utils import AlignmentLevel


@dataclass
class AlignerOutput(ModelOutput):
    alignments: List[Alignment] = None
    src_tokens: List[List[str]] = None
    tgt_tokens: List[List[str]] = None
    level: AlignmentLevel = None


class WordAligner:

    def __str__(self):
        raise NotImplementedError

    def align(self,
              src_sentences: List[str],
              tgt_sentences: List[str],
              src_lang_id: int = None,
              tgt_lang_id: int = None,
              ) -> AlignerOutput:
        raise NotImplementedError
