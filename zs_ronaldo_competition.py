import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def rectify(a):
    if(a<0):
        return 0.0
    else:
        return a

def area_bin(x):
    c=500
    if(x=='Left Side(L)'):
        c=50
    elif(x=='Left Side Center(LC)'):
        c=10
    elif(x=='Center(C)'):
        c=30
    elif(x=='Mid Ground(MG)'):
        c=100
    elif(x=='Right Side Center(RC)'):
        c=10
    elif(x=='Right Side(R)'):
        c=50
    return c

def score_region(x):
    c=0
    if(x=='Goal Line'):
        c=900
    elif(x=='Penalty Spot'):
        c=700
    elif(x=='Goal Area'):
        c=400
    elif(x=='Mid Range'):
        c=50
    elif(x=='Left Corner' or x=='Right Corner' or x=='Mid Ground Line'):
        c=30
    return c

def shot_type_bin(t):
    return int(t/10)

path='data.csv'
path_sub_sample = 'sample_submission.csv'
sub_sample = pd.read_csv(path_sub_sample)

sub_indices = sub_sample.iloc[:,0]

data = pd.read_csv(path)
cr7_data = data.iloc[:,:-5]

features = ['location_x','location_y','power_of_shot','distance_of_shot','area_of_shot','time_left','shot_basics']

cr7_data['location_x'] = cr7_data['location_x'].fillna(cr7_data['location_x'].mean())
cr7_data['location_y'] = cr7_data['location_y'].fillna(cr7_data['location_y'].mean())
cr7_data['power_of_shot'] = cr7_data['power_of_shot'].fillna(cr7_data['power_of_shot'].mean())
cr7_data['distance_of_shot'] = cr7_data['distance_of_shot'].fillna(cr7_data['distance_of_shot'].mean())
cr7_data['remaining_min'] = cr7_data['remaining_min'].fillna(cr7_data['remaining_min'].mean())
cr7_data['remaining_sec'] = cr7_data['remaining_sec'].fillna(cr7_data['remaining_sec'].mean())
cr7_data['type_of_shot'] = cr7_data['type_of_shot'].fillna('       0')
cr7_data['type_of_combined_shot'] = cr7_data['type_of_combined_shot'].fillna('       0')

cr7_data['location_y'] = [i*100 for i in cr7_data['location_y']]
cr7_data['type_of_shot']=cr7_data['type_of_shot'].apply(lambda x: int(x[7:]))
cr7_data['type_of_combined_shot']=cr7_data['type_of_combined_shot'].apply(lambda x: int(x[7:]))

cr7_data['type_of_shot'] = cr7_data['type_of_shot'].fillna(0)
cr7_data['type_of_combined_shot'] = cr7_data['type_of_combined_shot'].fillna(0)

cr7_data['shot_type'] = cr7_data['type_of_shot'] + cr7_data['type_of_combined_shot']
cr7_data['shot_type'] = cr7_data['shot_type'].apply(lambda x: shot_type_bin(x))

cr7_data['area_of_shot'] = cr7_data['area_of_shot'].apply(lambda x : area_bin(x))
cr7_data['is_goal'] = cr7_data['is_goal'].fillna(-1)

cr7_data['time_left'] = cr7_data['remaining_min'].apply(lambda x: x*60)
cr7_data['time_left'] = cr7_data['time_left']+cr7_data['remaining_sec']

cr7_data['shot_basics']=cr7_data['shot_basics'].apply(lambda x: score_region(x))

cr7_data['is_goal']=[int(x) for x in cr7_data['is_goal']]

test_indices=sub_indices
train_indices=[i for i in range(1, cr7_data.shape[0]) if i not in sub_indices]

X=cr7_data[features].iloc[train_indices]
y=cr7_data['is_goal'].iloc[train_indices]

# #To check the performance and score
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
#
# c_model = LinearRegression()
# c_model.fit(X_train,y_train)
# y_pred = c_model.predict(X_test)
# y_pred = pd.Series(y_pred)
# y_pred = y_pred.apply(lambda x: rectify(x))
# mae = mean_absolute_error(y_test, y_pred)
# print('MAE: '+ str(mae))
# print('Score: '+ str(1/(1+mae)))

#For final submission
X_train = X
X_test = cr7_data[features].iloc[test_indices]

y_train = y
c_model=LinearRegression()
c_model.fit(X_train,y_train)
y_pred = c_model.predict(X_test)
y_final=pd.Series(y_pred,name='is_goal')
y_final = y_final.apply(lambda x: rectify(x))
pid=pd.Series(test_indices,name='shot_id_number')

pred=pd.concat([pid,y_final],axis=1)
print(pred)

pred.to_csv('Ronaldo_Predictions.csv',index=False)
