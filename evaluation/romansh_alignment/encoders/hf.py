import math
from typing import Union

import numpy as np
import torch
from transformers import PreTrainedModel

from evaluation.romansh_alignment.encoders import SentenceEncoder
from evaluation.romansh_alignment.utils import LayerAggregation


class HuggingfaceEncoder(SentenceEncoder):

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer,
                 aggregation: Union[str, LayerAggregation] = LayerAggregation.SINGLE,
                 ):
        self.model = model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = tokenizer
        self.aggregation = aggregation

    def _embed_tokens(self, text: str, lang_id: int = None) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        # If sequence length is longer than maximum of model, split inputs into overlapping chunks and superimpose the hidden states
        max_length = self.model.config.max_position_embeddings - 2
        if max_length % 2 != 0:
            max_length -= 1
        # Make sure that input is padded to a multiple of max_length
        multiple = math.ceil(inputs['input_ids'].shape[1] / max_length) * max_length
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding="max_length",
            max_length=multiple,
        ).to(self.model.device)
        assert inputs['input_ids'].shape[1] == multiple
        assert inputs['input_ids'].shape[1] % max_length == 0
        chunks = self._chunk_inputs(inputs, max_length=max_length)
        chunk_outputs = []
        for chunk in chunks:
            with torch.no_grad():
                model_args = {
                    "output_hidden_states": True,
                    "return_dict": True,
                }
                if lang_id is not None:
                    model_args["lang_ids"] = torch.tensor([lang_id]).to(self.model.device)
                chunk_output = self.model(**chunk, **model_args)

                if self.aggregation == LayerAggregation.SINGLE:
                    chunk_output = chunk_output.last_hidden_state
                elif self.aggregation == LayerAggregation.AVERAGE:
                    chunk_output = torch.stack(chunk_output.hidden_states, dim=0).mean(dim=0)
                else:
                    raise ValueError(f"Invalid aggregation: {self.aggregation}")

                assert chunk_output.shape[1] == max_length
                chunk_output = chunk_output * chunk['attention_mask'].unsqueeze(-1)
                chunk_output = chunk_output.cpu().numpy()[0]
            chunk_outputs.append(chunk_output)
        embeddings = self._merge_outputs(inputs['input_ids'].shape[1], chunk_outputs)
        return embeddings

    def _chunk_inputs(self, inputs, max_length: int):
        """
        Split inputs into chunks of length `max_length` that overlap by 50%.
        """
        if inputs['input_ids'].shape[1] > max_length:
            chunks = []
            for i in range(0, inputs['input_ids'].shape[1], max_length // 2):
                chunk = {}
                for key, value in inputs.items():
                    chunk[key] = value[:, i:(i + max_length)]
                if chunk['input_ids'].shape[1] < max_length:
                    continue
                chunks.append(chunk)
        else:
            chunks = [inputs]
        return chunks

    def _merge_outputs(self, seq_len: int, chunk_outputs) -> np.ndarray:
        embeddings = np.zeros((seq_len, chunk_outputs[0].shape[1]))
        chunk_length = chunk_outputs[0].shape[0]
        assert len(chunk_outputs) * chunk_length == 2 * seq_len - chunk_length
        for i, chunk_output in enumerate(chunk_outputs):
            embeddings[(i * chunk_length // 2):(i * chunk_length // 2 + chunk_length)] += chunk_output
        # Average all overlapping parts
        embeddings[chunk_length // 2:-chunk_length // 2] /= 2
        return embeddings

    def _embed_sentence(self, text: str, lang_id: int = None) -> np.ndarray:
        token_embeddings = self.embed_tokens(text, lang_id)
        sentence_embeddings = np.mean(token_embeddings[token_embeddings.sum(axis=1) != 0], axis=0)
        return sentence_embeddings

    def __str__(self):
        return f"HuggingfaceEncoder({self.model.name_or_path.replace('/', '_')}, aggregation={self.aggregation})"
