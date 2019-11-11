# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


from fairseq import utils


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if 'net_input' in sample.keys():
            enc_seq_ids = sample['net_input']['src_tokens']
        else:
            # for decode step
            enc_seq_ids = sample['src_tokens']

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]

        is_copy = 'p_copy' in net_output[1].keys() and net_output[1]['p_copy'] is not None
        # print(net_output[1]['attn'])
        if is_copy:
            p_copy = net_output[1]['p_copy']

            enc_seq_ids = enc_seq_ids.unsqueeze(1).repeat(1, net_output[1]['attn'].size(1), 1)
            generate_prob = utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace) * (1-p_copy)
            copy_prob = net_output[1]['attn'] * p_copy
            final = generate_prob.scatter_add(2, enc_seq_ids, copy_prob)
            if log_probs:
                return torch.log(final + 1e-15)
            else:
                return final
        else:
            if log_probs:
                return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
            else:
                return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
