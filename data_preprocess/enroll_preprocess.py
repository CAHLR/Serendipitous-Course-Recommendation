__author__ = 'jwj'
import pandas as pd
import pickle
import numpy as np


# preprocess enrollment data, filter out courses enrolled less than 20 times
def preprocess():
    df = list()
    for i in range(2008, 2018):
        print(i)
        df.append(pd.read_csv('/research/EDW_enrollment_2007_2016/student_grade_data_'+str(i)+'.tsv', sep='\t', header=0, dtype={' Student Identifier(ppsk)': str, 'Section Nbr': str, 'Course Control Nbr': str}))
    data = pd.concat(df)
    data = data.loc[(data['Semester Year Name Concat']!='2008 Spring')&(data['Semester Year Name Concat']!='2008 Summer')&(data[' Student Identifier(ppsk)'].notnull())]
    data = data.loc[(data['Offering Type Desc']=='Primary')&(data['Grade Subtype Desc']!='Administrative Code')&(data['Grade Subtype Desc']!='Unknown')]
    data['Num_subject'] = data['Course Number'] + ' '+data['Course Subject Short Nm']
    data.drop(columns=['Course Number'], inplace=True)

    count = data.groupby('Num_subject')['Num_subject', 'Course Title Nm'].count()
    count = count.loc[count['Num_subject']>=20]
    data = data.loc[data['Num_subject'].isin(count.index), ['Num_subject', 'Course Subject Short Nm', 'Semester Year Name Concat', ' Student Identifier(ppsk)', 'undergraduate / graduate status', 'Instructor Name', 'Grade Type Desc', 'Grade Nm', 'Grade Points Nbr']]
    data.drop_duplicates(inplace=True)
    return data


# get student id, filter out students with enrollment records less than 5
# goes after preprocess()
def get_stu(data):
    count = data.groupby(' Student Identifier(ppsk)').size()
    count = pd.core.frame.DataFrame({'count': count}).reset_index()
    count.sort_values(by=['count'], ascending=False, inplace=True)
    count.reset_index(inplace=True)
    count = count.loc[:, [' Student Identifier(ppsk)', 'count']]
    count = count.loc[count['count'] >= 5]
    """
    dic1 = dict()
    li = list(count[' Student Identifier(ppsk)'])
    for i in li:
        dic1[i]=len(dic1)
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    alldata = {'stu_id': dic1, 'id_stu': reversed_dic1}
    f = open('stu_id.pkl', 'wb')
    pickle.dump(alldata, f)
    count.to_csv('stu_id.csv')
    print(count)
    """
    return count


# add student major to enrollment data
# goes after preprocess() and get_stu()
def add_major_to_data(data, count):
    data = data.loc[data[' Student Identifier(ppsk)'].isin(count[' Student Identifier(ppsk)'])]
    # print(data.loc[data[' Student Identifier(ppsk)']=='2862281'])
    major_data = pd.read_csv('/research/EDW_enrollment_2007_2016/student_majors_data.tsv', sep='\t', header=0, dtype={'ppsk': str, 'year.majors': str, 'term.majors': str})
    # print(major_data)
    #print(major_data.loc[major_data['ppsk']=='2862281'])
    #c = major_data.groupby('major')
    #print(c.groups.keys())
   # print(major_data.loc[major_data['major'].isnull()])
    major_data['Semester Year Name Concat'] = major_data['year.majors']+' '+major_data['term.majors']
    major_data.rename(index=str, columns={'ppsk': ' Student Identifier(ppsk)'}, inplace=True)
    major_data = major_data.loc[:, ['Semester Year Name Concat', ' Student Identifier(ppsk)', 'major']]
    data = pd.merge(data, major_data, how='left', on=['Semester Year Name Concat', ' Student Identifier(ppsk)'])
    #print(data.loc[data[' Student Identifier(ppsk)']=='2862281'])
    #print(data)
    #c = data.loc[data['major'].isnull(), ['Num_subject', 'undergraduate / graduate status', 'Semester Year Name Concat', 'major']]
   # c = c.groupby('undergraduate / graduate status').count()
    #print(c)
   # print(data)
    return data


# goes after add_major_to_data() because have to only left majors that are corrospond to enrollent records.
def get_major(data):
    count = data.groupby('major').size()
    #print(count)
    count = pd.core.frame.DataFrame({'count': count}).reset_index()
    #print(count)
    count.sort_values(by=['count'], ascending=False, inplace=True)
    count.reset_index(inplace=True)
    count = count.loc[:, ['major', 'count']]
    dic1 = dict()
    li = list(count['major'])
    for i in li:
        dic1[i]=len(dic1)
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    alldata = {'major_id': dic1, 'id_major': reversed_dic1}
    f = open('major_id.pkl', 'wb')
    pickle.dump(alldata, f)
    count.to_csv('major_id.csv')
    print(count)
    return count


# get ppsk semester major table
# goes after add_major_to_data() and get_major()
def get_enroll_seq_major(data):
    data = data.loc[:, [' Student Identifier(ppsk)', 'Semester Year Name Concat', 'Num_subject', 'major']]
    data.drop_duplicates(inplace=True)
    # order by semester to fill the nan as the previous one
    li = ['2008 Fall']
    for i in range(2009, 2018):
        for j in ['Spring', 'Summer', 'Fall']:
            li.append(str(i)+' '+j)
    data['Semester Year Name Concat'] = pd.Categorical(data['Semester Year Name Concat'], li)
    data.sort_values(by=[' Student Identifier(ppsk)', 'Semester Year Name Concat'], ascending=True, inplace=True)
    data.fillna(method='ffill', inplace=True)
    #data.to_csv('stu_sem_major.csv')
    # transfer to np matrix
    stu = pd.read_csv('stu_id.csv')
    sem = 28  # num of semesters
    mat_data = [[None for i in range(sem)] for j in range(stu.shape[0])]
    f = open('stu_id.pkl', 'rb')
    stu_id = pickle.load(f)['stu_id']
    f = open('semester_id.pkl', 'rb')
    sem_id = pickle.load(f)['semester_id']
    f = open('major_id.pkl', 'rb')
    major_id = pickle.load(f)['major_id']
    f = open('../course_preprocess/course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    num_course = len(course_id)
    f.close()
    data = data.groupby([' Student Identifier(ppsk)', 'Semester Year Name Concat'])
    for keys in data.groups.keys():
        print(data.get_group(keys))
        courses = list()
        data1 = data.get_group(keys).groupby('Num_subject')
        for keys1 in data1.groups.keys():
            print(data1.get_group(keys1))
            course1 = [course_id[keys1]]
            for i in data1.get_group(keys1)['major']:
                course1 += [major_id[i] + num_course]
            courses += [course1,]
        print(courses)
        mat_data[stu_id[keys[0]]][sem_id[keys[1]]] = courses
    mat_file = {'stu_sem_course&major': mat_data}
    f = open('stu_sem_course&major.pkl', 'wb')
    pickle.dump(mat_file, f)
    f.close()
    print(mat_data)


# get the whole data
# goes after preprocess() and get_stu()
def get_enroll_seq(data, count):
    data = data.loc[data[' Student Identifier(ppsk)'].isin(count[' Student Identifier(ppsk)']), ['Num_subject', 'Semester Year Name Concat', ' Student Identifier(ppsk)']]
    # transfer to np matrix
    stu = pd.read_csv('stu_id.csv')
    sem = 28  # num of semesters
    mat_data = [[None for i in range(sem)] for j in range(stu.shape[0])]
    f = open('stu_id.pkl', 'rb')
    stu_id = pickle.load(f)['stu_id']
    #id_stu = pickle.load(f)['id_stu']
    f = open('semester_id.pkl', 'rb')
    sem_id = pickle.load(f)['semester_id']
   # id_sem = pickle.load(f)['id_semester']
    f = open('../course_preprocess/course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    #id_course = pickle.load(f)['id_course']
    f.close()
    #print(data.loc[data['Num_subject']==id_course[1020]])
    data = data.groupby([' Student Identifier(ppsk)', 'Semester Year Name Concat'])
    for keys in data.groups.keys():
        print(data.get_group(keys))
        courses = tuple()
        for i in data.get_group(keys)['Num_subject']:
            courses = courses + (course_id[i],)
        print(courses)
        mat_data[stu_id[keys[0]]][sem_id[keys[1]]] = courses
    mat_file = {'stu_sem_course': mat_data}
    f = open('stu_sem_course.pkl', 'wb')
    pickle.dump(mat_file, f)
    f.close()
    print(mat_data)


def set_semester_id():
    li = ['2008 Fall']
    for i in range(2009, 2018):
        for j in ['Spring', 'Summer', 'Fall']:
            li.append(str(i)+' '+j)
    dic1 = dict()
    for sem in li:  # put the list(tuple) into a dictionary, the value is the ID of a word
        dic1[sem] = len(dic1)
    print(dic1)
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    print(reversed_dic1)
    alldata = {'semester_id': dic1, 'id_semester': reversed_dic1}
    f = open('semester_id.pkl', 'wb')
    pickle.dump(alldata, f)
    f.close()


# goes after preprocess() and get_stu(), get all instructors
def get_all_instructor(data, count):
    data = data.loc[data[' Student Identifier(ppsk)'].isin(count[' Student Identifier(ppsk)'])]
    data = data.loc[data['Instructor Name'].notnull()].reset_index()
    #a = data[data['Instructor Name'].str.contains('Zachary Pardos', na=False)]
    #a = a.groupby('Semester Year Name Concat').groups.keys()
    #print(a)
   # print(data.loc[(data['Semester Year Name Concat']=='2014 Spring')&(data['Num_subject']=='98 Env Sci, Policy, & Mgmt')&(data[' Student Identifier(ppsk)']=='1030904')])
    s = data['Instructor Name'].str.split('; ').apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = 'Instructor Name'
    del data['Instructor Name']
    data = data.join(s)
    #a = data.loc[data['Instructor Name']=='Zachary Pardos']
    #print(a)
    count = data.groupby('Instructor Name').size()
    count = pd.core.frame.DataFrame({'count_course': count}).reset_index()
    count.sort_values(by=['count_course'], ascending=False, inplace=True)
    count.reset_index(inplace=True)
    count = count.loc[:, ['Instructor Name', 'count_course']]
    sem_instr = data.loc[:, ['Instructor Name', 'Semester Year Name Concat']]
    sem_instr.drop_duplicates(inplace=True)
    count1 = sem_instr.groupby('Instructor Name').size()
    count1 = pd.core.frame.DataFrame({'count_sem': count1}).reset_index()
    count1.sort_values(by=['count_sem'], ascending=False, inplace=True)
    count1.reset_index(inplace=True)
    count1 = count1.loc[:, ['Instructor Name', 'count_sem']]

    instr = pd.merge(count, count1, on='Instructor Name')
    instr.to_csv('instructor_id.csv')


# goes after get_all_instructor, set threshold to semesters and times instructor has taught
def get_instructor(x, y):
    data = pd.read_csv('instructor_id.csv')   
    data = data.loc[(data['count_course']>=x)&(data['count_sem']>=y)]
    data.sort_values(by=['count_course', 'count_sem'], ascending=False, inplace=True)
    print(data)
    
    #print(data.loc[data['Instructor Name']=='Michael Jordan'])
    dic1 = dict()
    li = list(data['Instructor Name'])
    for i in li:
        dic1[i]=len(dic1)
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    alldata = {'instructor_id': dic1, 'id_instructor': reversed_dic1}
    f = open('instructor_id.pkl', 'wb')
    pickle.dump(alldata, f)


# get enrollment sequence with instructor, goes after preprocess() get_stu() get_all_instructor() and get_instructor()
def get_enroll_seq_instructor(data, count):
    data = data.loc[data[' Student Identifier(ppsk)'].isin(count[' Student Identifier(ppsk)'])]
    data = data.loc[data['Instructor Name'].notnull()].reset_index()
    s = data['Instructor Name'].str.split('; ').apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = 'Instructor Name'
    del data['Instructor Name']
    data = data.join(s)
    data = data.loc[:, ['Num_subject', 'Semester Year Name Concat', ' Student Identifier(ppsk)', 'Instructor Name']]
     # transfer to np matrix
    stu = pd.read_csv('stu_id.csv')
    sem = 28  # num of semesters
    mat_data = [[None for i in range(sem)] for j in range(stu.shape[0])]
    f = open('stu_id.pkl', 'rb')
    stu_id = pickle.load(f)['stu_id']
    f = open('semester_id.pkl', 'rb')
    sem_id = pickle.load(f)['semester_id']
    f = open('../course_preprocess/course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    num_course = len(course_id)
    f = open('instructor_id.pkl', 'rb')
    instrucor_id = pickle.load(f)['instructor_id']
    f.close()

    data = data.groupby([' Student Identifier(ppsk)', 'Semester Year Name Concat'])
    for keys in data.groups.keys():  # for each ppsk and semester
        courses = list()
        data1 = data.get_group(keys).groupby('Num_subject')
        for keys1 in data1.groups.keys():
            print(data1.get_group(keys1))
            course1 = [course_id[keys1]]
            for k in data1.get_group(keys1)['Instructor Name']:
                if k in instrucor_id.keys():
                    course1 = course1 + [instrucor_id[k]+num_course]
                else:
                    continue
            courses = courses + [course1,]
        print(courses)
        mat_data[stu_id[keys[0]]][sem_id[keys[1]]] = courses
    mat_file = {'stu_sem_course&instructor': mat_data}
    f = open('stu_sem_course&instructor.pkl', 'wb')
    pickle.dump(mat_file, f)
    f.close()
    print(mat_data)


# goes after preprocess() and get_stu()
def get_subject(data, count):
    data = data.loc[data[' Student Identifier(ppsk)'].isin(count[' Student Identifier(ppsk)'])]
    data = data.loc[data['Course Subject Short Nm'].notnull()].reset_index()
    count = data.groupby('Course Subject Short Nm').size()
    count = pd.core.frame.DataFrame({'count_subject': count}).reset_index()
    count.sort_values(by=['count_subject'], ascending=False, inplace=True)
    count.reset_index(inplace=True)
    count = count.loc[:, ['Course Subject Short Nm', 'count_subject']]
    count.to_csv('subject_id.csv')
    dic = dict()
    li = list(count['Course Subject Short Nm'])
    for i in li:
        dic[i] = len(dic)
    reversed_dic = dict(zip(dic.values(), dic.keys()))
    alldata = {'subject_id': dic, 'id_subject': reversed_dic}
    f = open('subject_id.pkl', 'wb')
    pickle.dump(alldata, f)


# goes after preprocess() get_stu() and get_subject()
def get_enroll_seq_subject(data, count):
    data = data.loc[data[' Student Identifier(ppsk)'].isin(count[' Student Identifier(ppsk)'])]
    stu = pd.read_csv('stu_id.csv')
    sem = 28  # num of semesters
    mat_data = [[None for i in range(sem)] for j in range(stu.shape[0])]
    f = open('stu_id.pkl', 'rb')
    stu_id = pickle.load(f)['stu_id']
    f = open('semester_id.pkl', 'rb')
    sem_id = pickle.load(f)['semester_id']
    f = open('../course_preprocess/course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    f = open('subject_id.pkl', 'rb')
    subject_id = pickle.load(f)['subject_id']
    f.close()
    num_course = len(course_id)
    data = data.groupby([' Student Identifier(ppsk)', 'Semester Year Name Concat'])
    for keys in data.groups.keys():
        print(data.get_group(keys))
        courses = list()
        for index, rows in data.get_group(keys).iterrows():
            i = rows['Num_subject']
            j = rows['Course Subject Short Nm']
            courses = courses + [[course_id[i], subject_id[j]+num_course]]
        print(courses)
        mat_data[stu_id[keys[0]]][sem_id[keys[1]]] = courses
    mat_file = {'stu_sem_course&subject': mat_data}
    f = open('stu_sem_course&subject.pkl', 'wb')
    pickle.dump(mat_file, f)
    f.close()
    print(mat_data)


# goes after preprocess() get_stu() get_subject() get_instructor()
def get_enroll_seq_ins_sub(data, count):
    data = data.loc[data[' Student Identifier(ppsk)'].isin(count[' Student Identifier(ppsk)'])]
    data = data.loc[data['Instructor Name'].notnull()].reset_index()
    s = data['Instructor Name'].str.split('; ').apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = 'Instructor Name'
    del data['Instructor Name']
    data = data.join(s)
    data = data.loc[:, ['Num_subject', 'Course Subject Short Nm', 'Semester Year Name Concat', ' Student Identifier(ppsk)', 'Instructor Name']]
    stu = pd.read_csv('stu_id.csv')
    sem = 28  # num of semesters
    mat_data = [[None for i in range(sem)] for j in range(stu.shape[0])]
    f = open('stu_id.pkl', 'rb')
    stu_id = pickle.load(f)['stu_id']
    f = open('semester_id.pkl', 'rb')
    sem_id = pickle.load(f)['semester_id']
    f = open('../course_preprocess/course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    f = open('subject_id.pkl', 'rb')
    subject_id = pickle.load(f)['subject_id']
    f = open('instructor_id.pkl', 'rb')
    instructor_id = pickle.load(f)['instructor_id']
    f.close()
    num_course = len(course_id)
    num_subject = len(subject_id)
    data = data.groupby([' Student Identifier(ppsk)', 'Semester Year Name Concat'])
    for keys in data.groups.keys():  # for each ppsk and semester
        courses = list()
        data1 = data.get_group(keys).groupby(['Num_subject', 'Course Subject Short Nm'])
        for keys1 in data1.groups.keys():
            print(data1.get_group(keys1))
            course1 = [course_id[keys1[0]], subject_id[keys1[1]]+num_course]
            for k in data1.get_group(keys1)['Instructor Name']:
                if k in instructor_id.keys():
                    course1 = course1 + [instructor_id[k]+num_course+num_subject]
                else:
                    continue
            courses = courses + [course1,]
        print(courses)
        mat_data[stu_id[keys[0]]][sem_id[keys[1]]] = courses
    mat_file = {'stu_sem_course&instructor&subject': mat_data}
    f = open('stu_sem_course&instructor&subject.pkl', 'wb')
    pickle.dump(mat_file, f)
    f.close()
    print(mat_data)



def test(data, count):
    data = data.loc[data[' Student Identifier(ppsk)'].isin(count[' Student Identifier(ppsk)']), ['Num_subject', 'Semester Year Name Concat', ' Student Identifier(ppsk)', 'Instructor Name']]
    # transfer to np matrix
    stu = pd.read_csv('stu_id.csv')
    sem = 28  # num of semesters
    mat_data = [[None for i in range(sem)] for j in range(stu.shape[0])]
    f = open('stu_id.pkl', 'rb')
    #stu_id = pickle.load(f)['stu_id']
    id_stu = pickle.load(f)['id_stu']
    f = open('semester_id.pkl', 'rb')
    #sem_id = pickle.load(f)['semester_id']
    id_sem = pickle.load(f)['id_semester']
    f = open('../course_preprocess/course_id.pkl', 'rb')
    #course_id = pickle.load(f)['course_id']
    id_course = pickle.load(f)['id_course']
    f.close()
    print(data.loc[(data['Semester Year Name Concat']==id_sem[19])&(data['Num_subject']==id_course[1020])])


def test1(data, count):
    data = data.loc[data[' Student Identifier(ppsk)'].isin(count[' Student Identifier(ppsk)']), ['Num_subject', 'Semester Year Name Concat', ' Student Identifier(ppsk)', 'Instructor Name', 'Grade Nm']]
    data = data.groupby(['Grade Nm'])
    print(data.groups.keys())


if __name__ == '__main__':
    #set_semester_id()
    data1 = preprocess()
    #count1 = get_stu(data1)
    #get_enroll_seq_ins_sub(data1, count1)
    #print(count1)
   # test1(data1, count1)
    #get_enroll_seq_instructor(data1, count1)
    #get_all_instructor(data1, count1)
   # get_instructor(100, 4)
    #get_enroll_seq(data1, count1)
    #data1 = add_major_to_data(data1, count1)
    #get_major(data1)
    #get_enroll_seq_major(data1)
