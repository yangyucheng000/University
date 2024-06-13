import mindspore as ms
from mindspore import nn, ops, Tensor
from typing import Dict, List, Optional, Tuple
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    gen_parser_from_dataclass,
)


class BaseFairseqModel(nn.Cell):
    """Base class for fairseq models."""
    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc(), delete_default=True)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")


    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]


    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif ops.is_tensor(net_output):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits = net_output.float()
            if log_probs:
                ls = nn.LogSoftmax(axis=-1)
                return ls(logits)
            else:
                ls = nn.Softmax(axis=-1)
                return ls(logits)
        raise NotImplementedError


    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, "")


    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += "."

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)



def check_type(module, expected_type):
    if hasattr(module, "unwrapped_module"):
        assert isinstance(
            module.unwrapped_module, expected_type
        ), f"{type(module.unwrapped_module)} != {expected_type}"
    else:
        assert isinstance(module, expected_type), f"{type(module)} != {expected_type}"


class FairseqEncoderModel(BaseFairseqModel):
    """Base class for encoder-only models.

    Args:
        encoder (FairseqEncoder): the encoder
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        check_type(self.encoder, FairseqEncoder)


    def construct(self, src_tokens, src_lengths, **kwargs):
        """
                Run the forward pass for a encoder-only model.

                Feeds a batch of tokens through the encoder to generate features.

                Args:
                    src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
                    src_lengths (LongTensor): source sentence lengths of shape `(batch)`

                Returns:
                    the encoder's output, typically of shape `(batch, src_len, features)`
                """
        return self.encoder(src_tokens, src_lengths, **kwargs)


    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output["encoder_out"]
        if ops.is_tensor(encoder_out):
            logits = encoder_out.astype(ms.float32)
            if log_probs:
                ls = nn.LogSoftmax(axis=-1)
                return ls(logits)
            else:
                ls = nn.Softmax(axis=-1)
                return ls(logits)
        raise NotImplementedError


    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions()





