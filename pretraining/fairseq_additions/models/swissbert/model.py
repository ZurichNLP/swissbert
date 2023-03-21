import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch
from fairseq.models import register_model_architecture, register_model
from fairseq.models.roberta import base_architecture
from fairseq.models.xmod import XMODModel
from omegaconf import DictConfig

from fairseq_additions.models.swissbert.hub_interface import SwissBERTHubInterface


@register_model("swissbert")
class SwissBERTModel(XMODModel):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="sentencepiece",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return SwissBERTHubInterface(x["args"], x["task"], x["models"][0])

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        """
        After loading a pre-trained XMOD model, initialize new language adapters, prune unneeded language adapters, and freeze some components
        """
        logging.info("Disabling strict loading from pre-trained XMOD model")
        if getattr(self.args, "train_new_embeddings", False):
            # Rename weights that depend on vocab size to avoid size mismatch
            state_dict["encoder.sentence_encoder.embed_tokens.weight_old"] = state_dict["encoder.sentence_encoder.embed_tokens.weight"]
            del state_dict["encoder.sentence_encoder.embed_tokens.weight"]
            state_dict["encoder.lm_head.weight_old"] = state_dict["encoder.lm_head.weight"]
            del state_dict["encoder.lm_head.weight"]
            state_dict["encoder.lm_head.bias_old"] = state_dict["encoder.lm_head.bias"]
            del state_dict["encoder.lm_head.bias"]
        super().load_state_dict(state_dict, strict=False, model_cfg=model_cfg, args=args)
        self._init_languages()
        if getattr(self.args, "prune_languages", False):
            self._prune_languages()
        self._freeze_shared_components()
        if getattr(self.args, "train_new_embeddings", False):
            self._initialize_new_embeddings(state_dict)

    def _init_languages(self):
        # Format of argument: de_DE->de_CH,fr_XX->fr_CH,de_DE+fr_XX->rm_CH
        if getattr(self.args, "init_languages", None) is None:
            return
        for mapping in self.args.init_languages.split(","):
            sources, target = mapping.split("->")
            sources = sources.split("+")
            logging.info(f"Initializing language adapter for {target} with language adapters for {sources}")
            for source in sources:
                assert source in self.model_languages, f"Source language {source} not in model languages"
            assert target in self.train_languages, f"Target language {target} not in train_languages"
            # Average state dicts of source adapters
            for k, v in self.encoder.sentence_encoder.layers._modules.items():
                average_state_dict = v.adapter_modules[sources[0]].state_dict()
                for source in sources[1:]:
                    for k_, v_ in v.adapter_modules[source].state_dict().items():
                        average_state_dict[k_] += v_
                for k_ in average_state_dict:
                    average_state_dict[k_] /= len(sources)
                v.adapter_modules[target].load_state_dict(average_state_dict)

    def _prune_languages(self):
        """
        Remove language adapters that are not trained
        """
        for i, (k, v) in enumerate(self.encoder.sentence_encoder.layers._modules.items()):
            for lang in self.model_languages:
                if lang not in self.train_languages:
                    if i == 0:
                        logging.info(f"Removing language adapter for {lang}")
                    del v.adapter_modules[lang]

    def _freeze_shared_components(self):
        """
        Freeze everything except language adapters
        """
        logging.info("❄Freezing everything except language adapters❄")
        for parameter in self.parameters():
            parameter.requires_grad = False
        for k, v in self.encoder.sentence_encoder.layers._modules.items():
            if hasattr(v, "adapter_layer_norm"):
                for parameter in v.adapter_layer_norm.parameters():
                    parameter.requires_grad = True
            for parameter in v.adapter_modules.parameters():
                parameter.requires_grad = True

    def _initialize_new_embeddings(self, pretrained_state_dict):
        """
        Initialize the new embeddings with the pre-trained embeddings of identical subwords
        Unfreeze the embeddings, including the positional embeddings
        """
        device = self.encoder.sentence_encoder.embed_tokens.weight.device
        dtype = self.encoder.sentence_encoder.embed_tokens.weight.dtype
        from transformers import XLMRobertaTokenizer
        logging.info("Initializing new embeddings with pre-trained embeddings")
        old_word_embeddings = pretrained_state_dict["encoder.sentence_encoder.embed_tokens.weight_old"]
        old_vocab_path = Path(self.args.old_vocab_path)
        assert old_vocab_path.exists() 
        old_vocab = XLMRobertaTokenizer(old_vocab_path).get_vocab()
        logging.info("Pre-trained vocabulary size: %d", len(old_vocab))
        assert len(old_vocab) == old_word_embeddings.shape[0]
        new_word_embeddings = self.encoder.sentence_encoder.embed_tokens.weight
        new_vocab_path = Path(self.args.new_vocab_path)
        assert new_vocab_path.exists()
        new_vocab = XLMRobertaTokenizer(new_vocab_path).get_vocab()
        logging.info("New vocabulary size: %d", len(new_vocab))
        assert len(new_vocab) == new_word_embeddings.shape[0]
        with torch.no_grad():
            num_overlap = 0
            for subword, new_index in new_vocab.items():
                if subword in old_vocab:
                    old_index = old_vocab[subword]
                    new_word_embeddings[new_index] = old_word_embeddings[old_index]
                    num_overlap += 1
            logging.info("Number of overlapping subwords: %d", num_overlap)
            # Tie LM head again
            if not self.args.untie_weights_roberta:
                self.encoder.lm_head.weight = new_word_embeddings
        logging.info("Unfreezing embeddings")
        for parameter in self.encoder.sentence_encoder.embed_tokens.parameters():
            parameter.requires_grad = True
        for parameter in self.encoder.sentence_encoder.embed_positions.parameters():
            parameter.requires_grad = True
        for parameter in self.encoder.lm_head.parameters():
            parameter.requires_grad = True
        self.to(device=device, dtype=dtype)


@register_model_architecture("swissbert", "swissbert_base")
def swissbert_base(args):
    args.ffn_modules = getattr(args, "ffn_modules", False)
    args.adapter_modules = getattr(args, "adapter_modules", True)
    args.adapter_layer_norm = getattr(args, "adapter_layer_norm", False)
    args.adapter_reuse_layer_norm = getattr(args, "adapter_reuse_layer_norm", True)
    args.ln_before_adapter = getattr(args, "ln_before_adapter", True)
    args.languages = getattr(
        args,
        "languages",
        [
            # 1. Language adapters of X-MOD
            "en_XX",
            "id_ID",
            "vi_VN",
            "ru_RU",
            "fa_IR",
            "sv_SE",
            "ja_XX",
            "fr_XX",
            "de_DE",
            "ro_RO",
            "ko_KR",
            "hu_HU",
            "es_XX",
            "fi_FI",
            "uk_UA",
            "da_DK",
            "pt_XX",
            "no_XX",
            "th_TH",
            "pl_PL",
            "bg_BG",
            "nl_XX",
            "zh_CN",
            "he_IL",
            "el_GR",
            "it_IT",
            "sk_SK",
            "hr_HR",
            "tr_TR",
            "ar_AR",
            "cs_CZ",
            "lt_LT",
            "hi_IN",
            "zh_TW",
            "ca_ES",
            "ms_MY",
            "sl_SI",
            "lv_LV",
            "ta_IN",
            "bn_IN",
            "et_EE",
            "az_AZ",
            "sq_AL",
            "sr_RS",
            "kk_KZ",
            "ka_GE",
            "tl_XX",
            "ur_PK",
            "is_IS",
            "hy_AM",
            "ml_IN",
            "mk_MK",
            "be_BY",
            "la_VA",
            "te_IN",
            "eu_ES",
            "gl_ES",
            "mn_MN",
            "kn_IN",
            "ne_NP",
            "sw_KE",
            "si_LK",
            "mr_IN",
            "af_ZA",
            "gu_IN",
            "cy_GB",
            "eo_EO",
            "km_KH",
            "ky_KG",
            "uz_UZ",
            "ps_AF",
            "pa_IN",
            "ga_IE",
            "ha_NG",
            "am_ET",
            "lo_LA",
            "ku_TR",
            "so_SO",
            "my_MM",
            "or_IN",
            "sa_IN",
            # 2. Added language adapters
            "de_CH",
            "fr_CH",
            "it_CH",
            "rm_CH",
        ],
    )
    base_architecture(args)
