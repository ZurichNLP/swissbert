import logging
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import sentencepiece as spm
from transformers import XLMRobertaTokenizer


def tokenize_xlm(
        txt_path: Path,
        out_path: Path,
):
    """
    Use Huggingface tokenizers to tokenize with XLM vocabulary
    """
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    with open(txt_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            bpe_tokens = tokenizer.tokenize(line)
            f_out.write(" ".join(bpe_tokens) + "\n")


def tokenize_hf(
        spm_model_path: Path,
        txt_path: Path,
        out_path: Path,
):
    """
    Tokenization with the SwissBERT SentencePiece model using Huggingface tokenizers
    """
    tokenizer = XLMRobertaTokenizer(spm_model_path)
    tokenizer.add_tokens(
        new_tokens=["<s>", "</s>", "<medium>", "<year>", "<month>"],
        special_tokens=True,
    )
    print(list(tokenizer.get_vocab())[:10])
    with open(txt_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            bpe_tokens = tokenizer.tokenize(line)
            f_out.write(" ".join(bpe_tokens) + "\n")


def tokenize_spm(
        spm_model_path: Path,
        txt_path: Path,
        out_path: Path,
        num_threads: int = 1,
    ):
    """
    Implementation that uses sentencepiece directly instead of HF tokenizers
    Not used due to different behavior w.r.t. special tokens
    """
    assert spm_model_path.exists()
    sp = spm.SentencePieceProcessor()
    sp.Init(
        model_file=str(spm_model_path),
        out_type=str,
        # add_bos=True,
        # add_eos=True,
        num_threads=num_threads,
    )
    with open(txt_path) as f_in, open(out_path, "w") as f_out:
        while True:
            lines = f_in.readlines(10_000)
            if not lines:
                break
            bpe_lines = sp.Encode(lines)
            for bpe_tokens in bpe_lines:
                f_out.write(" ".join(bpe_tokens) + "\n")


def create_spm_vocabulary(
        txt_paths: List[Path],
        name: str,
        sampling_alpha: float = 0.3,
        vocab_size: int = 50260,
        user_defined_symbols: List[str] = None,
        tmp_dir: Path = None,
        subsampling_ratio: float = 1.,
):
    for path in txt_paths:
        assert path.exists()
    if tmp_dir is not None:
        tmp_dir = Path(tmp_dir)
        assert tmp_dir.exists()

    logging.info("Counting lines")
    num_lines_orig = []
    # https://stackoverflow.com/a/9631635/3902795
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b
    for path in txt_paths:
        num_lines = 0
        with open(path, "r", encoding="utf-8", errors='ignore') as f:
            num_lines += sum(bl.count("\n") for bl in blocks(f))
        num_lines_orig.append(num_lines / 2)

    num_lines_orig = np.array(num_lines_orig)
    p_orig = num_lines_orig / num_lines_orig.sum()
    p_smooth = p_orig ** sampling_alpha
    p_smooth /= p_smooth.sum()
    num_lines_smooth = p_smooth / max(p_smooth) * max(num_lines_orig)
    expected_repetitions = num_lines_smooth / num_lines_orig
    assert (expected_repetitions >= 1).all()

    logging.info(f"Number of articles per original file: {num_lines_orig}")
    logging.info(f"Original proportions: {p_orig}")
    logging.info(f"Smoothened probabilities: {p_smooth}")
    logging.info(f"Number of articles smoothened: {num_lines_smooth}")
    logging.info(f"Expected repetitions: {expected_repetitions}")

    expected_repetitions *= subsampling_ratio
    logging.info(f"Expected repetitions after subsampling: {expected_repetitions}")

    tmp_in = tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir)
    logging.info(f"Writing lines to {tmp_in.name}")
    for txt_path, rep in zip(txt_paths, expected_repetitions):
        num_lines = 0
        with open(txt_path) as f:
            for line in f:
                if not line.strip():
                    continue
                full, remainder = divmod(rep, 1)
                for _ in range(int(full)):
                    tmp_in.write(line)
                    num_lines += 1
                if np.random.rand() < remainder:
                    tmp_in.write(line)
                    num_lines += 1
        print(f"{num_lines}, ", end="")
    print()

    spm.SentencePieceTrainer.Train(
        f'--user_defined_symbols={",".join(user_defined_symbols) if user_defined_symbols is not None else ""} '
        f'--input={tmp_in.name} '
        '--input_format=text '
        f'--model_prefix={name} '
        f'--vocab_size={vocab_size} '
        '--num_threads=40 '
        '--train_extremely_large_corpus=true '
        '--input_sentence_size=10000000 '
        '--shuffle_input_sentence=true '
    )

    tmp_in.close()
    Path(tmp_in.name).unlink()
