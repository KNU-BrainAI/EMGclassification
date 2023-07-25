from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



#%% multi file
dir_path =r'\\VOLKAN-MAZLUM\Users\VOLKAN MAZLUM\Desktop\Kore\s_1_angles'
file_pathlist = []
dir_list = []
file_type = []
for (root, directories, files) in os.walk(dir_path):
    for d in directories:
        d_path = os.path.join(root,d)
        dir_list.append(d)
        print(d)
        
    for file in files:
        file_path = os.path.join(root,file)
        file_pathlist.append(file_path)
        if ("E3" in file):
            file_type.append("E3")
        elif ("E2" in file):
            file_type.append("E2")
        elif ("E1" in file):
            file_type.append("E1")
        print(file_path)


train = pd.DataFrame(columns=['emg','stimulus','repetition','subject','rerepetition','restimulus','laterality','age','circumference'])
test = pd.DataFrame(columns=['emg','stimulus','repetition','subject','rerepetition','restimulus','laterality','age','circumference'])



now = 2000;
start=0;
end = 0;
idx=-1;
for j,file_path in enumerate(file_pathlist):
    res = loadmat(file_path)
    for i in range(len(res['emg'])-1):
        if(res['stimulus'][i][0] != now):
            now = res['stimulus'][i][0]
            start= i
        if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (2,5))):
            idx+=1
            train.loc[idx,'emg']=res['emg'][start:i]
            if (file_type[j]=="E3"):
                now = now+29
            elif (file_type[j]=="E2"):
                now = now+12
            train.loc[idx,'stimulus']=now-1
            train.loc[idx,'repetition']= f'{res["repetition"][i][0]}'
            train.loc[idx,'subject'] = f'subject{int(j/3)+1}'    
            train.loc[idx,'rerepetition']= f'{res["rerepetition"][i][0]}'
            train.loc[idx,'restimulus']= f'{res["restimulus"][i][0]-1}'  
            train.loc[idx,'laterality']= f'{res["laterality"]}'
            train.loc[idx,'age']= f'{res["age"]}'
            train.loc[idx,'circumference']= f'{res["circumference"]}'
        
train.to_pickle('ninaprodb44train.pkl')

now = 2000;
start=0;
end = 0;
idx=-1;
for j,file_path in enumerate(file_pathlist):
    res = loadmat(file_path)
    for i in range(len(res['emg'])-1):
        if(res['stimulus'][i][0] != now):
            now = res['stimulus'][i][0]
            start= i
        if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (1,3,4,6) )):
            idx+=1
            test.loc[idx,'emg']=res['emg'][start:i]
            if (file_type[j]=="E3"):
                now = now+29
            elif (file_type[j]=="E2"):
                now = now+12
            test.loc[idx,'stimulus']=now-1
            test.loc[idx,'repetition']= f'{res["repetition"][i][0]}'
            test.loc[idx,'subject'] = f'subject{int(j/3)+1}'    
            test.loc[idx,'rerepetition']= f'{res["rerepetition"][i][0]}'
            test.loc[idx,'restimulus']= f'{res["restimulus"][i][0]-1}'    
            test.loc[idx,'laterality']= f'{res["laterality"]}'
            test.loc[idx,'age']= f'{res["age"]}'
            test.loc[idx,'circumference']= f'{res["circumference"]}'
        

test.to_pickle('ninaprodb44test.pkl')