from pathlib import Path

import numpy as np
from cached_property import cached_property
from sqlitedict import SqliteDict


class SentenceEncoder:

    def embed_tokens(self, text: str, lang_id: int = None) -> np.ndarray:
        token_embeddings = self.cache.get(f"[TOKENS {lang_id}]" + text, None)
        if token_embeddings is None:
            token_embeddings = self._embed_tokens(text, lang_id)
            self.cache[f"[TOKENS {lang_id}]" + text] = token_embeddings
        return token_embeddings

    def _embed_tokens(self, text: str, lang_id: int = None) -> np.ndarray:
        raise NotImplementedError

    def embed_sentence(self, text: str, lang_id: int = None) -> np.ndarray:
        sentence_embeddings = self.cache.get(f"[SENTENCE {lang_id}]" + text, None)
        if sentence_embeddings is None:
            sentence_embeddings = self._embed_sentence(text, lang_id)
            self.cache[f"[SENTENCE {lang_id}]" + text] = sentence_embeddings
        return sentence_embeddings

    def _embed_sentence(self, text: str, lang_id: int = None) -> np.ndarray:
        token_embeddings = self.embed_tokens(text, lang_id)
        sentence_embeddings = np.mean(token_embeddings, axis=0)
        return sentence_embeddings

    def __str__(self):
        raise NotImplementedError

    @cached_property
    def cache_path(self) -> Path:
        cache_dir = Path(__file__).parent.parent / "cache"
        if not cache_dir.exists():
            cache_dir.mkdir()
        return cache_dir / f"{self}.sqlite"

    @cached_property
    def cache(self):
        return SqliteDict(self.cache_path, autocommit=True)
