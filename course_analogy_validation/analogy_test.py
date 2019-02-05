__author__ = 'jwj'
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from numpy import linalg as la



class Net(torch.nn.Module):
    def __init__(self, vdim, vector_dim):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(vdim, vector_dim, bias=False)
        self.l2 = torch.nn.Linear(vector_dim, vdim, bias=False)
    def forward(self, x):
        h = self.l1(x)
        y_pred = self.l2(h)
        return y_predi


def check_sim(word1, word2, word3, word4, dictionary, reverse_dic, weight):
    word = weight[dictionary[word3]] - weight[dictionary[word1]] + weight[dictionary[word2]]  # word is a vector
    #print(word[np.newaxis, :].shape)
    sim = cosine_similarity(word[np.newaxis, :], weight)[0]
    ## sim = get_sim(word, weight)
    nearest = (-sim).argsort()
    #print(nearest)
    #print(nearest)
    i = 0
    while reverse_dic[nearest[i]] == word1 or reverse_dic[nearest[i]] == word2 or reverse_dic[nearest[i]] == word3:
        i += 1
    if reverse_dic[nearest[i]] == word4:
        print(reverse_dic[nearest[i]], 'right')
        return 1
    else:
        print(reverse_dic[nearest[i]], 'wrong')
        return 0


if __name__ == '__main__':
    a = torch.load('../model/torch_model_128.pkl', map_location=lambda storage, loc: storage)
    layer = 1  # 1,2 or 3 (concat 1 and 2)
    param = a.state_dict()
    weight1 = param['l1.weight'].cpu().numpy().transpose()[:7487]
    weight2 = param['l2.weight'].cpu().numpy()
    if layer == 1:
        weight = weight1
    elif layer == 2:
        weight = weight2
    else:
        weight = np.concatenate((weight1, weight2), axis=1)
    #weight0 = la.norm(weight, 2, axis=1)
    #weight = weight / weight0[:, np.newaxis]
    course_file = open('../../course_preprocess/course_id.pkl', 'rb')
    course = pickle.load(course_file)
    course_id = course['course_id']
    vocab_size = len(course_id)
    id_course = course['id_course']
    #major_file = open('../../enroll_preprocess/major_id.pkl', 'rb')
    #major = pickle.load(major_file)
    #major_id = major['major_id']
    #id_major = major['id_major']
    course_file.close()

    vocab_size = len(id_course)

    analogy = pd.read_csv('../../course2vec/course_analogy_validation/analogy(andrew).txt', sep=',', names=list('abcd'))
    print(analogy)
    row_num = analogy.count(axis=0)['a']
    j = 0  # right predictions of each category
    f = 0.0  # num of each category
    k = 0  # total right predictions
    s = 0.0  # all num
    fi = open('analogy_result_128-a.txt', 'a')
    fi.write('test layer '+str(layer)+'\n')
    for i in range(row_num):
        if pd.isnull(analogy.loc[i, 'c']):
            #fl = open('analogy_result_512.txt', 'a')
            print("yes")
            try:
                print(j/f)
                fi.write(str(j)+'/'+str(f)+'='+str(j/f)+'\n')
            except ArithmeticError:
                print("start")
            a = 'Accuray of'+str(analogy.loc[i, 'a'])+':\n'
            print(a)
            fi.write(a)
            #fl.close()
            k += j
            j = 0
            f = 0
        else:
            #print('No')
            course1 = analogy.loc[i, 'a']
            course2 = analogy.loc[i, 'b']
            course3 = analogy.loc[i, 'c']
            course4 = analogy.loc[i, 'd']
            if course1 in course_id.keys() and course2 in course_id.keys() and course3 in course_id.keys() and course4 in course_id.keys():
                print(course1, course2, course3, course4)
                j += check_sim(course1, course2, course3, course4, course_id, id_course, weight)
                f += 1.0
                s += 1.0
    print(j/f)
    fi.write(str(j/f))
    b = '\nTotal accuracy is'+str(k)+'/'+str(s)+'='+str(k/s)+'\n'
    print(b)
    fi.write(b)
    fi.close()





