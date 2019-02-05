__author__ = 'jwj'
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torch


class Net(torch.nn.Module):
    def __init__(self, vdim, vector_dim):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(vdim, vector_dim, bias=False).cuda()
        self.l2 = torch.nn.Linear(vector_dim, vdim, bias=False).cuda()
    def forward(self, x):
        h = self.l1(x)
        y_pred = self.l2(h)
        return y_pred

def test(i, j, weight, course_id):
    valid_word1 = course_id[i]
    valid_word2 = course_id[j]
    sim = cosine_similarity(weight[valid_word1][np.newaxis, :], weight)[0]
    pos = list((-sim).argsort()[1:]).index(valid_word2)
    print(pos)
    return pos

if __name__ == '__main__':
    label = 3
    vali_pairs = pd.read_csv('vali_pairs.csv')
    rank = []
    my_model = torch.load('../model/torch_model_32.pkl')
    print(my_model)
    param = my_model.state_dict()
    #print(my_model.get_weights()[0]-my_model.get_weights()[1].T)
    #1/0
    if label == 1:
        print(1)
        weight = param['l1.weight'].cpu().numpy().transpose()[:7487]
        print(weight)
    elif label == 2:
        print(2)
        weight = param['l2.weight'].cpu().numpy()
    else:
        print(3)
        weight = np.concatenate((param['l1.weight'].cpu().numpy().transpose()[:7487], param['l2.weight'].cpu().numpy()), axis=1)
    f = open('../../course_preprocess/course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    for i, j in zip(vali_pairs['1'], vali_pairs['2']):
        print(i+" "+j)
        rank.append(test(i, j, weight, course_id))
    s = pd.Series(rank)
    print('torch_model_32-'+str(label))
    print(s.describe())
