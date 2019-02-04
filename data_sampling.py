_author__ = 'jwj'
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
import numpy as np
import pickle


window_size = 10  # the window of words around the target word that will be used to draw the context words from

course_file = open('../course_preprocess/course_id.pkl', 'rb')
course = pickle.load(course_file)
course_id = course['course_id']
vocab_size = len(course_id)
id_course = course['id_course']
course_file.close()
f = open('../enroll_preprocess/stu_sem_course&instructor&subject.pkl', 'rb')
data = pickle.load(f)['stu_sem_course&instructor&subject']
data = np.array(data)
couples = []
labels = []
print('Start sampling')
for i in range(data.shape[0]):
    #i = 102742
    print("this is"+str(i))
    a = data[i][data[i] != np.array(None)]
    if a.size==0:
        continue
    seq = np.sum(a)
    couple, label = skipgrams(seq, vocab_size, window_size=window_size, negative_samples=0)
    couples.extend(couple)
    #print(couples)
    print(len(couples))
    #print(couples)
    labels.extend(label)
course_target, course_context = zip(*couples)  # zip(*list[[]]) = unzip to tuple
print('Finish sampling')

all_data = {'course_target': course_target, 'course_context': course_context, 'labels': labels}
f = open('sampled_data_10.pkl','wb')
pickle.dump(all_data, f)
f.close()
