# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.parameter import Parameter
from .modeling_utils import (WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig, PreTrainedModel,
                             prune_linear_layer, add_start_docstrings)

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(PretrainedConfig):
    r"""
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

BertLayerNorm = nn.LayerNorm
BertGroupNorm = nn.GroupNorm

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.groups = config.groups
        #aself.batch_size = batch_size
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.ModuleList([nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0) for _ in range(self.groups)])
        self.position_embeddings = nn.ModuleList([nn.Embedding(config.max_position_embeddings, config.hidden_size) for _ in range(self.groups)])
        self.token_type_embeddings = nn.ModuleList([nn.Embedding(config.type_vocab_size, config.hidden_size) for _ in range(self.groups)])

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
#        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = BertGroupNorm(self.groups, self.groups * config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        # Shape of 'input_ids' : (M, B, L)
        m, b, l = input_ids.size()
        # seq_length = L, the last dim of 'input_ids'
        seq_length = input_ids.size(-1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Apply linear layer for each group
        # (M, B, L) -> (M, B, L, C)
        words_embeddings = torch.cat([self.word_embeddings[i](input_ids[i]).unsqueeze(0) for i in range(self.groups)])
        position_embeddings = torch.cat([self.position_embeddings[i](position_ids[i]).unsqueeze(0) for i in range(self.groups)])
        token_type_embeddings = torch.cat([self.token_type_embeddings[i](token_type_ids[i]).unsqueeze(0) for i in range(self.groups)])

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        # Change 'embeddings' dimension
        # (M, B, L, C) -> (B * L, M * C)
        embeddings = embeddings.view(self.groups, -1, self.hidden_size)
        embeddings = embeddings.transpose(0, 1).reshape(-1, self.groups * self.hidden_size)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # (B * L, M * C) -> (M, B, L, C) 
        embeddings = embeddings.reshape(-1, self.groups, self.hidden_size).transpose(0, 1)
        embeddings = embeddings.view(self.groups, b, -1, self.hidden_size)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.groups = config.groups
        #self.batch_size = batch_size
        self.hidden_size = config.hidden_size

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        """
        self.query = nn.ModuleList([nn.Linear(config.hidden_size, self.all_head_size) for _ in range(self.groups)])
        self.key = nn.ModuleList([nn.Linear(config.hidden_size, self.all_head_size) for _ in range(self.groups)])
        self.value = nn.ModuleList([nn.Linear(config.hidden_size, self.all_head_size) for _ in range(self.groups)])
        """

        self.query_weight = Parameter(torch.empty(self.groups, config.hidden_size, self.all_head_size))
        self.query_bias = Parameter(torch.empty(self.groups, 1, self.all_head_size))
        self.key_weight = Parameter(torch.empty(self.groups, config.hidden_size, self.all_head_size))
        self.key_bias = Parameter(torch.empty(self.groups, 1, self.all_head_size))
        self.value_weight = Parameter(torch.empty(self.groups, config.hidden_size, self.all_head_size))
        self.value_bias = Parameter(torch.empty(self.groups, 1, self.all_head_size))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # X shape: (M, B * L, C) -> (M, B, L, H, H_C), where C = H * H_C
#new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        new_x_shape = (self.groups, self.batch_size, -1, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # (M, B, L, H, H_C) -> (M, B, H, L, H_C)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        mixed_query_layer = torch.cat([self.query[i](hidden_states[i]).unsqueeze(0) for i in range(self.groups)])
        mixed_key_layer = torch.cat([self.key[i](hidden_states[i]).unsqueeze(0) for i in range(self.groups)])
        mixed_value_layer = torch.cat([self.value[i](hidden_states[i]).unsqueeze(0) for i in range(self.groups)])
        """

        # (M, B, L, C) -> (M, B * L, C)
        m, b, l, c = hidden_states.size()
        self.batch_size = b

        hidden_states = hidden_states.reshape(self.groups, -1, self.hidden_size)

        mixed_query_layer = torch.baddbmm(self.query_bias, hidden_states, self.query_weight)
        mixed_key_layer = torch.baddbmm(self.key_bias, hidden_states, self.key_weight)
        mixed_value_layer = torch.baddbmm(self.value_bias, hidden_states, self.value_weight)

        # (M, B, L, C) -> (M, B, H, L, H_C), where C = H * H_C
        mixed_query_layer = self.transpose_for_scores(mixed_query_layer)
        mixed_key_layer = self.transpose_for_scores(mixed_key_layer)
        mixed_value_layer = self.transpose_for_scores(mixed_value_layer)

        # (M, B, H, L, H_C) -> (M * B * H, L, H_C)
        query_layer = mixed_query_layer.reshape((-1,) + mixed_query_layer.size()[-2:])
        key_layer = mixed_key_layer.reshape((-1,) + mixed_key_layer.size()[-2:])
        value_layer = mixed_value_layer.reshape((-1,) + mixed_value_layer.size()[-2:])
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2))
#        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # (M * B * H, L, L) -> (M, B, H, L, L)
        attention_scores = attention_scores.reshape(mixed_query_layer.size()[:-2] + attention_scores.size()[-2:])

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        batched_attention_probs = attention_probs.reshape((-1,) + attention_probs.size()[-2:])
        context_layer = torch.bmm(batched_attention_probs, value_layer)
#        context_layer = torch.matmul(batched_attention_probs, value_layer)
        context_layer = context_layer.reshape(attention_probs.size()[:-2] + context_layer.size()[-2:])

        # (M, B, H, L, H_C) -> (M, B, L, C)
        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.groups = config.groups
        #self.batch_size = batch_size
        self.hidden_size = config.hidden_size

        self.dense_weight = Parameter(torch.empty(self.groups, config.hidden_size, config.hidden_size))
        self.dense_bias = Parameter(torch.empty(self.groups, 1, config.hidden_size))

        self.LayerNorm = BertGroupNorm(self.groups, self.groups * config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        m, b, l, c = hidden_states.size()
        # (M, B, L, C) -> (M, B * L, C)
        hidden_states = hidden_states.reshape(self.groups, -1, self.hidden_size)
        input_tensor = input_tensor.reshape(self.groups, -1, self.hidden_size)

        hidden_states = torch.baddbmm(self.dense_bias, hidden_states, self.dense_weight)

        hidden_states = self.dropout(hidden_states)

        # Change 'embeddings' dimension
        # (M, B * L, C) -> (B * L, M * C)
        ln_input = hidden_states + input_tensor
        ln_input = ln_input.transpose(0, 1).reshape(-1, self.groups * self.hidden_size)
        hidden_states = self.LayerNorm(ln_input)
        
        # (B * L, M * C) -> (M, B, L, C)
        hidden_states = hidden_states.reshape(-1, self.groups, self.hidden_size).transpose(0,1)
        hidden_states = hidden_states.view(self.groups, b, -1, self.hidden_size)

        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.groups = config.groups
        #self.batch_size = batch_size
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.groups = config.groups
        #self.batch_size = batch_size
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

#        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_weight = Parameter(torch.empty(self.groups, config.hidden_size, config.intermediate_size))
        self.dense_bias = Parameter(torch.empty(self.groups, 1, config.intermediate_size))

        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        m, b, l, c = hidden_states.size()
        hidden_states = hidden_states.reshape(self.groups, -1, self.hidden_size)
        hidden_states = torch.baddbmm(self.dense_bias, hidden_states, self.dense_weight)
        hidden_states = hidden_states.reshape(self.groups, b, -1, self.intermediate_size)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.groups = config.groups
        #self.batch_size = batch_size
        self.hidden_size = config.hidden_size
#        self.dense_weight = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense_weight = Parameter(torch.empty(self.groups, config.intermediate_size, config.hidden_size))
        self.dense_bias = Parameter(torch.empty(self.groups, 1, config.hidden_size))

        self.LayerNorm = BertGroupNorm(self.groups, self.groups * config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # (M, B, L, C) -> (M, B * L, C)
        m,b,l,i = hidden_states.size()
        hidden_states = hidden_states.reshape(self.groups, -1, i)

        hidden_states = torch.baddbmm(self.dense_bias, hidden_states, self.dense_weight)
        
        hidden_states = self.dropout(hidden_states)

        # Change 'embeddings' dimension
        # (M, B * L, C) -> (B * L, M * C)
        input_tensor = input_tensor.reshape(self.groups, -1, self.hidden_size)
        ln_input = hidden_states + input_tensor
        ln_input = ln_input.transpose(0, 1).reshape(-1, self.groups * self.hidden_size)
        hidden_states = self.LayerNorm(ln_input)
        
        # (B * L, M * C) -> (M, B, L, C)
        hidden_states = hidden_states.reshape(-1, self.groups, self.hidden_size).transpose(0,1)
        hidden_states = hidden_states.view(self.groups, b, -1, self.hidden_size)

        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.groups = config.groups
        #self.batch_size = batch_size
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.groups = config.groups
        #self.batch_size = batch_size
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.groups = config.groups
#self.batch_size = batch_size
        self.hidden_size = config.hidden_size

        self.dense = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.groups)])
        self.dense_weight = Parameter(torch.empty(self.groups, config.hidden_size, config.hidden_size))
        self.dense_bias = Parameter(torch.empty(self.groups, 1, config.hidden_size))
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, :, 0]
        pooled_output = torch.cat([self.dense[i](first_token_tensor[i]).unsqueeze(0) for i in range(self.groups)])
        #pooled_output = torch.baddbmm(self.dense_bias, first_token_tensor, self.dense_weight)
        pooled_output = self.activation(pooled_output)

        return pooled_output

class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

#        self.apply(self.init_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [group_size, batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(2).unsqueeze(3)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
