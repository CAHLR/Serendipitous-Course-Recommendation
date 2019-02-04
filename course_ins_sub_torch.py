__author__ = 'jwj'
import torch
import numpy as np
import pandas as pd
import pickle
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

vector_dim = 300
sampled_data = 'sampled_data_10.pkl'


valid_size = 10     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.


class Net(torch.nn.Module):
    def __init__(self, idim, vector_dim, odim):
        super(Net, self).__init__()
       # self.weight1 = torch.cat((init_param['l1.weight'][:, :7487], torch.FloatTensor(np.zeros(([vector_dim, len(major_id)]))).cuda()), 1)
        #self.weight2 = init_param['l2.weight']
        self.l1 = torch.nn.Linear(idim, vector_dim, bias=False).cuda()
        self.l2 = torch.nn.Linear(vector_dim, odim, bias=False).cuda()
    def forward(self, x):
        h = self.l1(x)
        y_pred = self.l2(h)
        return y_pred


def weight_init(m):
    print(m)
    if type(m) == torch.nn.Linear:
        if m.in_features == indim - 1:
            m.weight.data = torch.cat((init_param['l1.weight'][:, :7487], torch.FloatTensor(np.zeros(([vector_dim, len(major_id)-1]))).cuda()), 1)
            print(m.weight.data)
        elif m.in_features == vector_dim:
            m.weight.data = init_param['l2.weight']



def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    index = np.random.randint(ylen, size=batch_size)
    target_batch = x[index]
    context_batch = y[index]
    #context_batch.astype(int, copy=True)
    target = np.zeros((batch_size, indim))
    for i in range(batch_size):
        target[i, target_batch[i]] = 1
    context_course = zip(*context_batch)
    context = list(context_course)[0]
   # print(context)
    #context = list(context)
    #context = [int(a) for a in context]
    return target, context


def train_batch():
    for t in range(10):  # epoch
        #scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        #scheduler.step()
        optimizer.zero_grad()
        for i in range(iter):
            target, context = generate_batch_data_random(word_target, word_context, batch_s)
            target = torch.FloatTensor(target)
            context = torch.LongTensor(context)
            target = Variable(target, requires_grad=False)
            target = target.cuda()
            context = Variable(context, requires_grad=False)
            context = context.cuda()
            y_pred = model(target).cuda()
            loss = loss_fn(y_pred, context)
            loss_fn.cuda()
            loss.backward()
            print('epoch' + str(t+1)+':' + 'The'+str(i+1)+'-th interation: training loss'+str(loss.data[0])+'\n')
            #optimizer.zero_grad()
        model.l1.weight.grad = model.l1.weight.grad / float(iter)
        model.l2.weight.grad = model.l2.weight.grad / float(iter)
        optimizer.step()

        if need_validation == True:
            validation_set_loss(t)


def train_mini_batch():
    for t in range(10):  # epoch
        #scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        #scheduler.step()
        for i in range(iter):
            target, context = generate_batch_data_random(word_target, word_context, batch_s)
            target = torch.FloatTensor(target)
            context = torch.LongTensor(context)
            target = Variable(target, requires_grad=False)
            target = target.cuda()
            context = Variable(context, requires_grad=False)
            context = context.cuda()
            optimizer.zero_grad()
            y_pred = model(target).cuda()
            loss = loss_fn(y_pred, context)
            loss_fn.cuda()
            print('epoch' + str(t+1)+':' + 'The'+str(i+1)+'-th interation: training loss'+str(loss.data[0])+'\n')
            loss.backward()
            optimizer.step()

        if need_validation == True:
            validation_set_loss(t)


# batch_size>16384, out of memory
def train_mini_batch1():
    for t in range(10):  # epoch
        #scheduler = MultiStepLR(optimizer, milestones=[5,6,7,8,9], gamma=0.1)
        #scheduler.step()
        for i in range(iter):
            optimizer.zero_grad()
            for j in range(iter1):
                target, context = generate_batch_data_random(word_target, word_context, 1024)
                target = torch.FloatTensor(target)
                context = torch.LongTensor(context)
                target = Variable(target, requires_grad=False)
                target = target.cuda()
                context = Variable(context, requires_grad=False)
                context = context.cuda()
                y_pred = model(target).cuda()
                loss = loss_fn(y_pred, context)
                loss_fn.cuda()
                loss.backward()
            print('epoch' + str(t+1)+':' + 'The'+str(i+1)+'-th interation: training loss'+str(loss.data[0])+'\n')
            model.l1.weight.grad = model.l1.weight.grad / float(iter1)
            model.l2.weight.grad = model.l2.weight.grad / float(iter1)
            optimizer.step()
        if need_validation == True:
            validation_set_loss(t)


def validation_set_loss(t):
    print('validation_loss_calculating')
    iter = vali_length // 4096
    print(iter)
    vali_loss_epoch = 0
    for i in range(iter+1):
        print(i)
        if i == iter:
            target_vali = word_target_vali[iter*4096:]
            context_vali = word_context_vali[iter*4096:]
        else:
            target_vali = word_target_vali[i:i+4096]
            context_vali = word_context_vali[i:i+4096]
        vali_batch_length = len(target_vali)
        target_vali_np = np.zeros((vali_batch_length, indim))
        for j in range(vali_batch_length):
            target_vali_np[j, target_vali[j]] = 1
        context_vali_np = zip(*context_vali)
        context_vali_np = list(context_vali_np)[0]
        target_vali_np = torch.FloatTensor(target_vali_np)
        context_vali_np = torch.LongTensor(context_vali_np)
        target_vali = Variable(target_vali_np, volatile=True).cuda()
        context_vali = Variable(context_vali_np, volatile=True).cuda()

        context_vali_pred = model(target_vali).cuda()
        vali_loss_iter = loss_fn(context_vali_pred, context_vali).cuda()
        vali_loss_epoch += vali_loss_iter.data[0] * vali_batch_length
    vali_loss_epoch /= vali_length
    print('epoch' + str(t+1)+': validation loss'+str(vali_loss_epoch)+'\n')
    f = open('validation_loss_32768-50epoch.txt', 'a')
    f.write(str(vali_loss_epoch)+',')
    f.close()




if __name__ == '__main__':
    mini_batch = 1  # 0: on batch, other: mini-batch
    need_validation = False
    batch_s = 32768
   # init = torch.load('../course2vec/model/torch_model_32768.pkl')
    #init_param = init.state_dict()

    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    course_file = open('../course_preprocess/course_id.pkl', 'rb')
    course = pickle.load(course_file)
    course_id = course['course_id']
    vocab_size = len(course_id)
    id_course = course['id_course']
    subject_file = open('../enroll_preprocess/subject_id.pkl', 'rb')
    subject = pickle.load(subject_file)
    subject_id = subject['subject_id']
    id_subject = subject['id_subject']
    instructor_file = open('../enroll_preprocess/instructor_id.pkl', 'rb')
    instructor = pickle.load(instructor_file)
    instructor_id = instructor['instructor_id']
    id_instructor = instructor['id_instructor']
    course_file.close()

    f = open(sampled_data, 'rb')
    data = pickle.load(f)
    f.close()
    word_target = list(data['course_target'])
    word_context = list(data['course_context'])
    word_target = np.array(word_target)
    word_context = np.array(word_context)
    print("construct model")
    indim = len(course_id) + len(subject_id) + len(instructor_id)
    outdim = len(course_id)
    data_length = len(word_target)

    if need_validation == True:
        # separate validation set and training set
        vali_length = data_length // 10
        word_target_vali = word_target[:vali_length]
        word_context_vali = word_context[:vali_length]
        word_target = word_target[vali_length:]
        word_context = word_context[vali_length:]
        vali_loss1 = list()

    loss_fn = torch.nn.CrossEntropyLoss()
    model = Net(indim, vector_dim, outdim)
    #model.apply(weight_init)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iter1 = batch_s//1024
    iter = len(word_target)//batch_s


    print(iter)
    if mini_batch == 0:
        print('train on batch')
        train_batch()
    elif mini_batch != 0 and batch_s <= 16384:
        print('train on mini_batch')
        train_mini_batch()
    elif mini_batch != 0 and batch_s > 16384:
        print('train on mini_batch>16384')
        train_mini_batch1()
    torch.save(model, 'torch_model_32768.pkl')
    a = torch.load('torch_model_32768.pkl')
    print(a)
    param = a.state_dict()
    print(param['l1.weight'])
    print(param['l2.weight'])

