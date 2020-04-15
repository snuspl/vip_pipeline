import os
import sys
import tensorflow as tf
import json
import numpy as np

from model_selector import Model as Model_Selector
from model_params import Config
from dataset_e import VTTExample
from utils_e import utils_file
from graph_handler import GraphHandler
from socket import *

path=os.getcwd()

_DECODE_BATCH_SIZE = 16
_JSON_TAG = 'json'

def _text2data(que, des, ans):
    data = {}
    data['que'] = que
    data['description'] = des
    data['ans'] = ans

    data_list = []
    data_list.append(data)

    return data_list

def _jsontext2data(text):

    try:
        data_list = json.loads(text)
    except Exception as ex:
        print("Error : ", ex)

    return data_list

def _file2data(file):

    return utils_file.read_file(file, _JSON_TAG)

def load_model(data, model_params):
    with tf.compat.v1.variable_scope('model') as scope:
        model = Model_Selector(data.emb_mat_token, data.emb_mat_glove, len(data.dicts['token']), len(data.dicts['char']),
                           data.max_token_size, data.max_ans_size, model_params, scope.name)

    graphHandler = GraphHandler(model, model_params)

    #gpu_options = tf.GPUOptions()
    graph_config = tf.ConfigProto()
    graph_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=graph_config)#config=graph_config)

    graphHandler.initialize(sess)

    return model, sess


def predict(data, model, sess, print_all=True):
    logits_list, predictions_list = [], []
    que_list, ans_list, des_list = [], [], []
    output_list = []
    L = []

    infer_time = 0
    post_time = 0
    for sample_batch, _, _, _ in data.generate_batch_sample():
        feed_dict = model.get_feed_dict(sample_batch, 'prediction')
        logits = sess.run(model.logits, feed_dict=feed_dict)

        predictions = np.argmax(logits, -1)
        logits_list.append(logits)
        predictions_list.append(predictions)

        questions = [sample['que'] for sample in sample_batch]
        descriptions = [sample['description'] for sample in sample_batch]
        answers = [[answer for answer in sample['ans']] for sample in sample_batch]

        que_list.append(questions)
        des_list.append(descriptions)
        ans_list.append(answers)

        if print_all:
            for logit, prediction, answer_candidate in zip(logits, predictions, answers):
                #tf.compat.v1.logging.info("Answer Confidence : {}\n".format(logit))
                #tf.compat.v1.logging.info("Selected Answer : {}\n".format(answer_candidate[prediction]))
                output_list.append({'Answer_Confidence' : logit, 'Selected_Answer' : answer_candidate[prediction]})

    #print('Answer Confidence : {}\n'.format(output_list[0]['Answer_Confidence']))
    #print('Selected Answer : {}\n'.format(output_list[0]['Selected_Answer']))

    return output_list[0]['Selected_Answer']

class AnsSelectModel():
    def __init__(self):
        # Model configuration before input
        model_params = Config(os.path.join(path, 'data'), os.path.join(path, 'model'))

        model_params.batch_size = _DECODE_BATCH_SIZE
        model_params.mode = 'prediction'

        model_params.load_model = True

        dicts = utils_file.read_file(model_params.dict_path, _JSON_TAG)
        
        encoded_data = VTTExample('prediction', model_params, dicts, False)
        model, sess = load_model(encoded_data, model_params)

        self.model = model
        self.sess = sess
        self.encoded_data = encoded_data

    def predict(self, input_data):
        model = self.model
        sess = self.sess
        encoded_data = self.encoded_data

        q_text = input_data["question"]
        d_text = input_data["a"]
        a_list = [input_data["c"], input_data["d"]]

        # Input to the model & predict
        data = _text2data(q_text, d_text, a_list)
        encoded_data.prepare_input(data)

        model.token_max_length = encoded_data.max_token_size

        final_answer = predict(encoded_data, model, sess)

        return [final_answer]

if __name__ == '__main__':
    q_text = "question sample"
    d_text = "description example"
    c = "answer1"
    d = "answer2"
    init_result = init()
    inference(init_result, {"question":q_text, "a": d_text, "c" : c, "d" : d})
