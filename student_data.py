#data = pd.read_csv("colour-data.csv")
#X = (data[['R','G','B']]/255).values
#y = data['Label'].values
#X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

dataset = pd.read_csv("slim-xAPI-Edu-Data.csv")
#students failed if they are in class L.
dataset['Failed'] = np.where(dataset['Class'] == 'L', True, False)
dataset['gender'] = np.where(dataset['gender']=='M',1,0)
dataset['Relation'] = np.where(dataset['Relation']=='Father',1,0)
dataset['ParentAnsweringSurvey'] = np.where(dataset['ParentAnsweringSurvey'] == 'Yes', 1, 0)
dataset['ParentschoolSatisfaction'] = np.where(dataset['ParentschoolSatisfaction'] == 'Yes', 1, 0)
dataset['AbsentMoreThanWeek'] = np.where(dataset['StudentAbsenceDays'] == 'Above-7', 1, 0)
dataset['Semester'] = np.where(dataset['Semester'] == 'F', 1, 0)
X = dataset[['raisedhands', 'VisITedResources', 'SectionID', 'Topic', 'StageID', 'AnnouncementsView', 'Semester', 'Discussion', 'gender', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'AbsentMoreThanWeek']].values
y = dataset[['Failed']].values

X_train, X_test, y_train, y_test = train_test_split(X, y)
    