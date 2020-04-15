from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random

import numpy as np
import torch
from torch .utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
#from prediction_model import BertConfig, BertForSequenceClassification
from modeling import BertConfig, BertForSequenceClassification
from optimization import BERTAdam

path = os.getcwd()
class InputExample(object):

    def __init__(self, guid, text_a, text_b, label = None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class LogicProcessor(object):
    def get_labels(self):
        return ['2','3']

    def _create_examples(self, text_a, text_b, set_type):

        guid = '%s' % (set_type)
        text_a = tokenization.convert_to_unicode(text_a)
        text_b = tokenization.convert_to_unicode(text_b)
        example = InputExample(guid, text_a, text_b)
        return example
def convert_id_to_label(label_,label_list) :
    label_map = {}
    for (i, labels) in enumerate(label_list):
        label_map[i] = labels
    label = label_[0]
    return label_map[label]
def convert_examples_to_features(example, label_list, max_seq_length, tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i


    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = tokenizer.tokenize(example.text_b)

    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = []
    segment_ids = []
    tokens.append('[CLS]')
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append('[SEP]')
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1]*len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if example.label == None:
        feature = InputFeatures(input_ids, input_mask, segment_ids)

    else:
        label_id = label_map[example.label]
        feature = InputFeatures(input_ids, input_mask, segment_ids, label_id)



    return feature

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length < max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class LogicLevelModel():
    def __init__(self):
        self.bert_config = BertConfig.from_json_file(os.path.join(path, 'uncased_L-12_H-768_A-12/bert_config.json'))
        self.max_sequence_length = 128

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.max_sequence_length > self.bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                    self.max_sequence_length, self.bert_config.max_position_embeddings))

        self.processor = LogicProcessor()
        self.label_list = self.processor.get_labels()
        self.tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(path, 'uncased_L-12_H-768_A-12/vocab.txt'),do_lower_case=False)

        self.model = BertForSequenceClassification(self.bert_config,len(self.label_list))
        init_checkpoint = os.path.join(path, 'model/logic_model_500.bin')
        #Future save model Load code

        if init_checkpoint is not None:
            self.model.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))

        self.model.to(self.device)


    def predict(self, text_a, text_b):
        device = self.device

        eval_example = self.processor._create_examples(text_a, text_b,'input example')
        eval_feature = convert_examples_to_features(eval_example, self.label_list, self.max_sequence_length, self.tokenizer)

        input_ids = torch.tensor([eval_feature.input_ids],dtype=torch.long)
        input_mask = torch.tensor([eval_feature.input_mask], dtype=torch.long)
        segment_ids = torch.tensor([eval_feature.segment_ids], dtype=torch.long)
        if eval_feature.label_id == None :
            label_ids = None
        else :
            label_ids = torch.tensor(eval_feature.label_ids, dtype=torch.long)

        #eval_data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        #eval_dataloader = DataLoader(eval_data)

        self.model.eval()

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        if label_ids == None:
            pass
        else :
            label_ids = label_ids.to(device)
        
        L = []
        if label_ids == None:
            logits = self.model(input_ids, segment_ids, input_mask, label_ids)
        else :
            loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids)

        logits = logits.detach().cpu().numpy()

        output = np.argmax(logits, axis=1)
        output = convert_id_to_label(output, self.label_list)

        return output





