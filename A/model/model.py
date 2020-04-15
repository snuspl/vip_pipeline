import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from pytorch_pretrained_bert import BertForQuestionAnswering
from pytorch_pretrained_bert import BertModel

from pytorch_pretrained_bert import BertConfig
import torch
from torch.nn.parameter import Parameter
import model.modeling_bert as modeling_bert
import model.modeling_bert_grouped as modeling_bert_grouped
import model.modeling_utils as modeling_utils

class FriendsBertModel(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        config = BertConfig(vocab_size_or_config_json_file=30522)
        config.output_hidden_states=True

        self.bert1 = modeling_bert.BertModel.from_pretrained('bert-base-cased')
        self.bert2 = modeling_bert.BertModel.from_pretrained('bert-base-cased')

        self.berts = [self.bert1, self.bert2]
        self.groups = 2

        self.fc = nn.Linear(1536, num_classes)


    def merge_berts(self):
        # Create an instance of grouped model
        config = self.berts[0].config
        config.groups = self.groups
        self.bert = modeling_bert_grouped.BertModel(config)
    
        # Copy paramters in embedding layers
        for i in range(self.groups):
            self.bert.embeddings.word_embeddings[i].weight = self.berts[i].embeddings.word_embeddings.weight
            self.bert.embeddings.position_embeddings[i].weight = self.berts[i].embeddings.position_embeddings.weight
            self.bert.embeddings.token_type_embeddings[i].weight = self.berts[i].embeddings.token_type_embeddings.weight

        self.bert.embeddings.LayerNorm.weight = Parameter(torch.cat([self.berts[i].embeddings.LayerNorm.weight for i in range(self.groups)]))
        self.bert.embeddings.LayerNorm.bias = Parameter(torch.cat([self.berts[i].embeddings.LayerNorm.bias for i in range(self.groups)]))

        # Copy parameters in encoder layers
        for i in range(len(self.bert.encoder.layer)):
            layer = self.bert.encoder.layer[i]
            # attention
            layer.attention.self.query_weight = Parameter(torch.cat([self.berts[j].encoder.layer[i].attention.self.query.weight.transpose(0,1).unsqueeze(0) for j in range(self.groups)]))
            layer.attention.self.query_bias = Parameter(torch.cat([self.berts[j].encoder.layer[i].attention.self.query.bias.unsqueeze(0).unsqueeze(0) for j in range(self.groups)]))
            layer.attention.self.key_weight = Parameter(torch.cat([self.berts[j].encoder.layer[i].attention.self.key.weight.transpose(0,1).unsqueeze(0) for j in range(self.groups)]))
            layer.attention.self.key_bias = Parameter(torch.cat([self.berts[j].encoder.layer[i].attention.self.key.bias.unsqueeze(0).unsqueeze(0) for j in range(self.groups)]))
            layer.attention.self.value_weight = Parameter(torch.cat([self.berts[j].encoder.layer[i].attention.self.value.weight.transpose(0,1).unsqueeze(0) for j in range(self.groups)]))
            layer.attention.self.value_bias = Parameter(torch.cat([self.berts[j].encoder.layer[i].attention.self.value.bias.unsqueeze(0).unsqueeze(0) for j in range(self.groups)]))

            layer.attention.output.dense_weight = Parameter(torch.cat([self.berts[j].encoder.layer[i].attention.output.dense.weight.transpose(0,1).unsqueeze(0) for j in range(self.groups)]))
            layer.attention.output.dense_bias = Parameter(torch.cat([self.berts[j].encoder.layer[i].attention.output.dense.bias.unsqueeze(0).unsqueeze(0) for j in range(self.groups)]))
            layer.attention.output.LayerNorm.weight = Parameter(torch.cat([self.berts[j].encoder.layer[i].attention.output.LayerNorm.weight for j in range(self.groups)]))
            layer.attention.output.LayerNorm.bias = Parameter(torch.cat([self.berts[j].encoder.layer[i].attention.output.LayerNorm.bias for j in range(self.groups)]))

            # intermediate
            layer.intermediate.dense_weight = Parameter(torch.cat([self.berts[j].encoder.layer[i].intermediate.dense.weight.transpose(0,1).unsqueeze(0) for j in range(self.groups)]))
            layer.intermediate.dense_bias = Parameter(torch.cat([self.berts[j].encoder.layer[i].intermediate.dense.bias.unsqueeze(0).unsqueeze(0) for j in range(self.groups)]))

            # output
            layer.output.dense_weight = Parameter(torch.cat([self.berts[j].encoder.layer[i].output.dense.weight.transpose(0,1).unsqueeze(0) for j in range(self.groups)]))
            layer.output.dense_bias = Parameter(torch.cat([self.berts[j].encoder.layer[i].output.dense.bias.unsqueeze(0).unsqueeze(0) for j in range(self.groups)]))
            layer.output.LayerNorm.weight = Parameter(torch.cat([self.berts[j].encoder.layer[i].output.LayerNorm.weight for j in range(self.groups)]))
            layer.output.LayerNorm.bias = Parameter(torch.cat([self.berts[j].encoder.layer[i].output.LayerNorm.bias for j in range(self.groups)]))

        # Copy parameters in pooler layers
        self.bert.pooler.dense_weight = Parameter(torch.cat([self.berts[j].pooler.dense.weight.transpose(0,1).unsqueeze(0) for j in range(self.groups)]))
        self.bert.pooler.dense_bias = Parameter(torch.cat([self.berts[j].pooler.dense.bias.unsqueeze(0).unsqueeze(0) for j in range(self.groups)]))
        for i in range(self.groups):
            self.bert.pooler.dense[i].weight = self.berts[i].pooler.dense.weight
            self.bert.pooler.dense[i].bias = self.berts[i].pooler.dense.bias

        # Delete original models
        del self.bert1
        del self.bert2

    def forward(self, input1, input2): #,input3,input4):
        inp = (torch.cat([input1[0].unsqueeze(0), input2[0].unsqueeze(0)]), torch.cat([input1[1].unsqueeze(0), input2[1].unsqueeze(0)]))
        x,_ = self.bert(inp[0], inp[1])
        x1, x2 = x

        #x1,_ = self.bert1(input1[0],input1[1])
        #x2,_ = self.bert2(input2[0],input2[1])
        #x3,_ = self.bert1(input3[0],input3[1])
        #x4,_ = self.bert2(input4[0],input4[1])
        y1 = torch.cat((x1[:,0],x2[:,0]),1)
        #y2 = torch.cat((x3[:,0],x4[:,0]),1)
        y1 = self.fc(y1)
        #y2 = self.fc(y2)
        y1 = F.softmax(y1,dim=1)
        #y2 = F.softmax(y2,dim=1)
        #print(y1,y2)
        #print(y1[:,0].shape,y2[:,0].shape)
        #return (y1,y2)
        return y1
