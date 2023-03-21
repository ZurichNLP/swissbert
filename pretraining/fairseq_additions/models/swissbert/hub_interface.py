from fairseq import utils
from fairseq.models.xmod import XMODHubInterface


class SwissBERTHubInterface(XMODHubInterface):

    def fill_mask(self, masked_input: str, topk: int = 5, **kwargs):
        """
        Source: https://github.com/facebookresearch/fairseq/blob/58cc6cca18f15e6d56e3f60c959fe4f878960a60/fairseq/models/roberta/hub_interface.py#L156
        Added **kwargs to allow passing lang_id
        """
        masked_token = "<mask>"
        assert (
            masked_token in masked_input and masked_input.count(masked_token) == 1
        ), "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(
            masked_token
        )

        text_spans = masked_input.split(masked_token)
        text_spans_bpe = (
            (" {0} ".format(masked_token))
            .join([self.bpe.encode(text_span.rstrip()) for text_span in text_spans])
            .strip()
        )
        tokens = self.task.source_dictionary.encode_line(
            "<s> " + text_spans_bpe + " </s>",
            append_eos=False,
            add_if_not_exist=False,
        )

        masked_index = (tokens == self.task.mask_idx).nonzero(as_tuple=False)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with utils.model_eval(self.model):
            features, extra = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
                **kwargs,
            )
        logits = features[0, masked_index, :].squeeze()
        prob = logits.softmax(dim=0)
        values, index = prob.topk(k=topk, dim=0)
        topk_predicted_token_bpe = self.task.source_dictionary.string(index)

        topk_filled_outputs = []
        for index, predicted_token_bpe in enumerate(
            topk_predicted_token_bpe.split(" ")
        ):
            predicted_token = self.bpe.decode(predicted_token_bpe)
            # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
            if predicted_token_bpe.startswith("\u2581"):
                predicted_token = " " + predicted_token
            if " {0}".format(masked_token) in masked_input:
                topk_filled_outputs.append(
                    (
                        masked_input.replace(
                            " {0}".format(masked_token), predicted_token
                        ),
                        values[index].item(),
                        predicted_token,
                    )
                )
            else:
                topk_filled_outputs.append(
                    (
                        masked_input.replace(masked_token, predicted_token),
                        values[index].item(),
                        predicted_token,
                    )
                )
        return topk_filled_outputs
