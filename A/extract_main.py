import os
path = os.getcwd()

import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from dataset.dataset_generation_testset import data_preprocessing,writefile,generate_samples
import json

class DescExtractModel():
    def __init__(self):
        model_path = os.path.join(path, 'saved/models/Friends_Bert/0831_172003/model_best.pth')

        args = argparse.ArgumentParser(description='Extraction Model')
        args.add_argument('-r', '--resume', default=model_path, type=str,help='starting model path when resume')
        args.add_argument('-d', '--device', default=0, type=str,help='indices of GPUs to enable')
        config = ConfigParser(args)

        # build model architecture
        model = config.initialize('arch', module_arch)

        checkpoint = torch.load(model_path,  map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        model.merge_berts()

        # Move model to gpu device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # load datasets
        dataset = data_preprocessing(os.path.join(path, 'dataset/'))
        tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False, do_lower_case =False)
        FileObject = open(os.path.join(path, "dataset/FriendsQA_desc.json"), 'r')
        QA_dataset = json.load(FileObject)

        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.QA_dataset = QA_dataset

    def predict(self, input_data):
        model = self.model
        dataset = self.dataset
        tokenizer = self.tokenizer
        QA_dataset = self.QA_dataset

        question = input_data["question"]
        ID = input_data["vid"]
        episode = ID.split('_')[0]
        scene = ID.split('_')[1]

        samples, num_samples = generate_samples(dataset, episode, question)

        data_loader = module_data.FriendsBertDataLoader(samples, num_samples, shuffle=False, validation_split=0.0, num_workers=0, tokenizer=tokenizer)

        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        L=[]
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                input_data = [data['input1'][0],data['input2'][0]]
                input_segment = [data['input1'][1],data['input2'][1]]
                
                for i in range(2):
                    input_data[i] = input_data[i].to(device)
                    input_segment[i] = input_segment[i].to(device)

                output = model((input_data[0],input_segment[0]),(input_data[1],input_segment[1]))
                L = output[:, 1:].cpu().numpy()

        # Find Scene w/ minimum output value
        L = list(L.flatten())
        min_idx = L.index(min(L))
        scene = samples['shot_id'].values[min_idx]

        shot_id = "000" # Change This!!
        query = scene + "_" + shot_id
        scene_description = ' '.join(QA_dataset[query]["desc"])

        return scene_description
