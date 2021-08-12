#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[5]:


df1 = pd.read_csv(r"C:\Users\ashok\Desktop\data\Bengaluru_House_DataPJ2.csv")
df1.head()


# In[6]:


df1.shape


# In[9]:


df1.columns


# In[11]:


df1.groupby('area_type')['area_type'].agg('count')


# In[14]:


# Drop the unwanted feature(attributes)
df2 =df1.drop(['area_type','society','availability','balcony'],axis='columns')
df2.head()


# In[21]:



# Data Cleaning: Handle NA values


# In[15]:


df2.isnull().sum()


# In[16]:


df3=df2.dropna()
df3.isnull().sum()


# In[17]:


df3.shape


# In[18]:


df3['size'].unique()


# In[22]:


# Feature Engineering
# Add new feature(integer) for bhk (Bedrooms Hall Kitchen)
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[23]:


df3.head()


# In[24]:


df3['bhk'].unique()


# In[25]:


df3[df3.bhk>20]


# In[26]:


df3.total_sqft.unique()


# In[27]:


def is_float(x):
    try:
        float(x)
    except:
     return False
    return True


# In[29]:


df3[~df3['total_sqft'].apply(is_float)].head()


# In[47]:


# def convert_sqft_to_num(x):
#     tokens = x.split('_')
#     if len(tokens) == 2:
#         return (float(tokens[0])+float(tokens[1]))/2
#     try:
#         return float(x)
#     except:
#         return None


# In[52]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[53]:


convert_sqft_to_num('2166')


# In[54]:


df4 =df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head(3)


# In[55]:


df4.loc[30]


# In[56]:


df4.head(4)


# # Feature Engineering

# ### Add new feature called price per square feet

# In[58]:


df5 =df4.copy()
df5['price_per_sqft'] = df5['price'] *100000/df5['total_sqft']
df5.head()


# In[59]:


df5.location.unique()


# In[60]:


len(df5.location.unique())


# In[62]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats


# In[64]:


len(location_stats[location_stats<=10])


# In[65]:


location_stats_less_than_10 =location_stats[location_stats<=10]
location_stats_less_than_10


# In[66]:


len(df5.location.unique())


# In[67]:


df5.location =df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[68]:


df5.head()


# In[69]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[70]:


df5.shape


# In[71]:


df6 =df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[72]:


df6.price_per_sqft.describe()


# In[77]:


# def remove_pps_outliers(df):
#     df_out = pd.DataFrame()
#     for key, subdf in df.groupby('location'):
#         m = np.mean(subdf.price_per_sqft)
#         st = np.std(subdf.price_per_sqft)
#         reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
#         df_out = pd.concat([df_out,reduced_df],ignore_index=True)
#         return df_out
    
# df7 = remove_pps_outliers(df6)    
# df7.shape
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# In[80]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()

plot_scatter_chart(df7,"Rajaji Nagar") 


# In[81]:


plot_scatter_chart(df7,"Hebbal")


We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.

{
    '1' : {
        'mean': 4000,
        'std: 2000,
        'count': 34
    },
    '2' : {
        'mean': 4300,
        'std: 2300,
        'count': 22
    },    
}
# In[82]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[83]:


plot_scatter_chart(df7,"Hebbal")


# In[85]:


import matplotlib
plt.figure(figsize=(20,10))
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel('Price Per Squre Feet')
plt.ylabel('count')


# In[86]:



# Outlier Removal Using Bathrooms Feature
df8.bath.unique()


# In[88]:


df8[df8.bath>10]


# In[89]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel('Number of bathrooms')
plt.ylabel('counts')


# In[90]:


df8[df8.bath>df8.bhk+2]


# In[92]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[93]:


df10=df9.drop(['size','price_per_sqft'],axis='columns')
df10.head()


# In[96]:


dummies=pd.get_dummies(df10.location)
dummies.head(4)


# In[97]:


df11 =pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[98]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# In[99]:



X = df12.drop(['price'],axis='columns')
X.head(3)


# In[101]:


y = df12.price
y.head()


# In[102]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[103]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[104]:


# Use K Fold cross validation to measure accuracy of our LinearRegression model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# In[105]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
                 }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[106]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[107]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[108]:



predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[109]:


predict_price('Indira Nagar',1000, 2, 2)


# In[110]:


predict_price('Indira Nagar',1000, 3, 3)


# In[111]:


# Export the tested model to a pickle file
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[112]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# # END

# In[ ]:




