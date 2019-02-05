__author__ = 'jwj'
import pandas as pd
import pickle

df = list()
for i in range(2008, 2018):
    print(i)
    df.append(pd.read_csv('/research/EDW_enrollment_2007_2016/student_grade_data_'+str(i)+'.tsv', sep='\t', header=0, dtype={'Section Nbr': str, 'Course Control Nbr': str}))
data = pd.concat(df)
#print(data.loc[data['Course Number']=='-'])
data = data.loc[(data['Offering Type Desc']=='Primary')&(data['Grade Subtype Desc']!='Administrative Code')&(data['Grade Subtype Desc']!='Unknown')&(data['Semester Year Name Concat']!='2008 Spring')&(data['Semester Year Name Concat']!='2008 Summer')]
data['Num_subject'] = data['Course Number'] + ' '+data['Course Subject Short Nm']
data.drop(columns=['Course Number', 'Course Subject Short Nm'], inplace=True)

count = data.groupby('Num_subject')['Num_subject', 'Course Title Nm'].count()
count = count.loc[count['Num_subject']>=20]
print(count)


data = data.loc[data['Num_subject'].isin(count.index), ['Num_subject', 'Crs Academic Dept Short Nm']]
count = data.groupby('Num_subject').size()
#print(type(count))
count = pd.core.frame.DataFrame({'count' :count}).reset_index()
count.sort_values(by=['count'], ascending=False, inplace=True)
count.reset_index(inplace=True)
count = count.iloc[:,[1,2]]
#print(count)
# get course id to dictionary
dic = count.to_dict('dict')
dic1 = dic['Num_subject']
reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
alldata = {'course_id': reversed_dic1, 'id_course':dic1}
f = open('course_id.pkl', 'wb')
pickle.dump(alldata, f)
f.close()
# testify whether Num_subject can be the key
data.drop_duplicates(subset={'Num_subject', 'Crs Academic Dept Short Nm'}, inplace=True)
#print(data)
data.to_csv('course_dept.csv', index=False)
count.to_csv('course_enroll_num.csv')

