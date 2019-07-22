
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import ensemble

def rectify(a):
    if(a<0):
        return 0.0
    else:
        return a

path='data.csv'
path_sub_sample = 'sample_submission.csv'
sub_sample = pd.read_csv(path_sub_sample)

sub_indices = sub_sample.iloc[:,0]
data = pd.read_csv(path)
cr7_data = data.iloc[:,:-5]

cr7_data['distance_of_shot'] = cr7_data['distance_of_shot'].apply(lambda x: x if x<50 or x==float('NaN') else 50)
cr7_data['distance_of_shot'] = cr7_data['distance_of_shot'].fillna(int(cr7_data['distance_of_shot'].mean()))

cr7_data['match_event_id'] = cr7_data['match_event_id'].fillna(cr7_data['match_event_id'].mean())

cr7_data['mid']=cr7_data['area_of_shot'].apply(lambda x: 1.0 if x=='Mid Ground(MG)' else 0.0)

cr7_data['penalty']=cr7_data['shot_basics'].apply(lambda x: 1.0 if x=='Penalty Spot' else 0)
cr7_data['goal area']=cr7_data['shot_basics'].apply(lambda x: 1.0 if x=='Goal Area' else 0)

cr7_data['location_x'] = cr7_data['location_x'].fillna(cr7_data['location_x'].mean())
cr7_data['location_y'] = cr7_data['location_y'].fillna(cr7_data['location_y'].mean())


test_indices=sub_indices
train_indices=[i for i in range(1, cr7_data.shape[0]) if i not in sub_indices]

cr7_data['is_goal'] = cr7_data['is_goal'].fillna(cr7_data['is_goal'].mean())

features=['distance_of_shot','location_y','location_x','match_event_id']


X=cr7_data[features].iloc[train_indices]
y=cr7_data['is_goal'].iloc[train_indices]

#To check the performance and score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
params = {'n_estimators': 350, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.009, 'loss': 'ls'}
c_model = ensemble.GradientBoostingRegressor(**params)
c_model.fit(X_train,y_train)
y_pred = c_model.predict(X_test)
y_pred = pd.Series(y_pred)
y_pred = y_pred.apply(lambda x: rectify(x))
mae = mean_absolute_error(y_test, y_pred)
print('MAE: '+ str(mae))
print('Score: '+ str(1/(1+mae)))

#For final submission

X_train = X
X_test = cr7_data[features].iloc[test_indices]
y_train = y
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
c_model = ensemble.GradientBoostingRegressor(**params)
c_model.fit(X_train,y_train)
y_pred = c_model.predict(X_test)
y_final=pd.Series(y_pred,name='is_goal')
y_final = y_final.apply(lambda x: rectify(x))
pid=pd.Series(test_indices,name='shot_id_number')

pred=pd.concat([pid,y_final],axis=1)
print(pred)

pred.to_csv('Ronaldo_Predictions3.csv',index=False)
