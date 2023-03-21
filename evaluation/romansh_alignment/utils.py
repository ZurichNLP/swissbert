from typing import List, Union, Tuple

import numpy as np
import torch
from nltk import Alignment
from transformers import BatchEncoding


class AlignmentLevel(str):
    WORD = "word"
    TOKEN = "token"


class WordToTokenStrategy(str):
    ALL_TOKENS = "all-tokens"
    FIRST_TOKEN = "first-token"


class TokenToWordStrategy(str):
    ANY_TOKEN = "any-token"
    ALL_TOKENS = "all-tokens"
    FIRST_TOKEN = "first-token"


class LayerAggregation(str):
    SINGLE = "single"
    AVERAGE = "average"


class WordAlignment(Alignment):

    @classmethod
    def fromstring(cls, s) -> 'WordAlignment':
        return WordAlignment(Alignment.fromstring(s))

    @classmethod
    def from_labels(cls, alignment_labels: np.ndarray) -> 'WordAlignment':
        pairs = {tuple(pair) for pair in list(zip(*alignment_labels.nonzero()))}
        return WordAlignment(pairs)

    @property
    def level(self):
        return AlignmentLevel.WORD


def subword_to_word_map(sentence: str, encoding: BatchEncoding) -> List[int]:
    if encoding.input_ids.shape[0] != 1:
        raise NotImplementedError("Only batch size 1 is supported")
    to_word_map = []
    word_offsets: List[Tuple[int, int]] = []
    for i, char in enumerate(sentence):
        if i == 0 or sentence[i - 1] == " ":
            word_offsets.append((i, i + 1))
        elif char != " ":
            word_offsets[-1] = (word_offsets[-1][0], i + 1)
    # Provided encoding.offsets is not accurate for custom vocabularies, need to correct
    subword_offsets: List[Tuple[int, int]] = []
    for i, (token_start, token_stop) in enumerate(encoding[0].offsets):
        if encoding[0].tokens[i].startswith("▁") and (token_stop - token_start) == len(encoding[0].tokens[i]):
            token_start += 1
        subword_offsets.append((token_start, token_stop))
    for token_start, token_stop in subword_offsets:
        if (token_start, token_stop) == (0, 0):
            to_word_map.append(None)
            continue
        for i, (word_start, word_stop) in enumerate(word_offsets):
            if token_start >= word_start and token_stop <= word_stop:
                word_index = i
                break
        to_word_map.append(word_index)
    words = sentence.split()
    try:
        assert len(set([i for i in to_word_map if i is not None])) == len(words)
        assert max([i for i in to_word_map if i is not None]) == len(words) - 1, print(sentence, to_word_map, words)
        assert len(to_word_map) == encoding.input_ids.size(-1)
    except AssertionError:
        raise ValueError(f"Failed to map subwords to words: {sentence} {to_word_map} {words}")
    return to_word_map


def alignment_error_rate(
        references: List[Alignment],
        hypotheses: List[Alignment],
):
    # Does not account for "possible" links
    assert len(references) == len(hypotheses)
    num_reference_pairs = 0
    num_hypothesis_pairs = 0
    num_pairs_in_intersection = 0
    for reference, hypothesis in zip(references, hypotheses):
        assert type(reference) is type(hypothesis)
        num_reference_pairs += len(reference)
        num_hypothesis_pairs += len(hypothesis)
        num_pairs_in_intersection += len(reference & hypothesis)
    aer = 1.0 - 2 * num_pairs_in_intersection / float(num_hypothesis_pairs + num_reference_pairs)
    return aer


def alignment_f1_score(
        references: List[Alignment],
        hypotheses: List[Alignment],
) -> Tuple[float, float, float]:
    # Does not account for "possible" links
    assert len(references) == len(hypotheses)
    num_reference_pairs = 0
    num_hypothesis_pairs = 0
    num_pairs_in_intersection = 0
    for reference, hypothesis in zip(references, hypotheses):
        assert type(reference) is type(hypothesis)
        num_reference_pairs += len(reference)
        num_hypothesis_pairs += len(hypothesis)
        num_pairs_in_intersection += len(reference & hypothesis)
    precision = num_pairs_in_intersection / float(num_hypothesis_pairs)
    recall = num_pairs_in_intersection / float(num_reference_pairs)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def bert_score(
        query_embeddings: np.ndarray,
        document_embeddings: Union[List[np.ndarray], np.ndarray],
        device: str = "cpu",
):
    """
    Adapted from https://github.com/Tiiiger/bert_score/blob/cb582ed5c88b02230b8f101173fd959b68023dc6/bert_score/utils.py#L469
    """
    assert document_embeddings[0].shape[-1] == query_embeddings.shape[-1]
    if isinstance(document_embeddings, list):
        # Pad document_embeddings to the same length with zeros
        max_length = max(len(embeddings) for embeddings in document_embeddings)
        document_embeddings = [np.pad(embeddings, ((0, max_length - len(embeddings)), (0, 0)), 'constant') for embeddings in document_embeddings]
        document_embeddings = np.array(document_embeddings)

    with torch.no_grad():
        ref_embedding = torch.from_numpy(query_embeddings).unsqueeze(0).repeat(len(document_embeddings), 1, 1).to(device)
        hyp_embedding = torch.from_numpy(document_embeddings).to(device)

        ref_masks = (ref_embedding != 0).all(-1)
        hyp_masks = (hyp_embedding != 0).all(-1)
        # Avoid NaN
        ref_embedding[~ref_masks] = 1
        hyp_embedding[~hyp_masks] = 1

        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

        batch_size = ref_embedding.size(0)
        sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
        masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
        masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

        masks = masks.float().to(sim.device)
        sim = sim * masks

        word_precision = sim.max(dim=2)[0]
        word_recall = sim.max(dim=1)[0]

        P = word_precision.sum(dim=1) / hyp_masks.sum(dim=1)
        R = word_recall.sum(dim=1) / ref_masks.sum(dim=1)
        F = 2 * P * R / (P + R)
        return F.cpu().numpy()
