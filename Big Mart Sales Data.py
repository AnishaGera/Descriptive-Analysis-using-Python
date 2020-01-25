#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


cd Downloads


# In[3]:


ls|grep csv


# In[4]:


file=pd.read_csv("Train.csv")


# In[5]:


file.columns


# In[6]:


file.head(70)


# In[7]:


file.describe()


# In[8]:


len(file)


# In[9]:


print((file.loc[0:10]))


# In[10]:


def find_datatype(data):
    distinct_elements=set()
    for i in range(len(data)):
        if data[i] not in distinct_elements:
            distinct_elements.add(data[i])
    no_of_distinct_elements= len(distinct_elements)
    print("no_of_distinct_elements",no_of_distinct_elements)
    if no_of_distinct_elements < 85:
        return "Categorical"
    else:
        return "Continous"


# In[11]:


list_items=file["Item_Identifier"].tolist()


# In[12]:


find_datatype(list_items)


# In[13]:


lis=file.columns
lis


# In[14]:


cols = ["Item_Identifier","Item_Weight",
              "Item_Fat_Content","Item_Visibility","Item_Type","Item_MRP",
             "Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size",
             "Outlet_Location_Type","Outlet_Type","Item_Outlet_Sales"]


# In[15]:


cols


# In[16]:


data_types=[]
for col in cols:
    list_items=file[col].tolist()
    data_types.append(find_datatype(list_items))
    
    


# In[17]:


data_types


# In[18]:


df=file[file.columns[1:]]
corr = df.corr(method="pearson")
corr


# In[19]:


file.apply(lambda x: sum(x.isnull()))


# In[20]:


file.apply(lambda x: len(x.unique()))


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


plt.hist(file['Item_MRP'],bins=10)


# In[23]:


file.plot(kind='scatter',x='Item_Weight',y='Item_MRP')


# In[24]:


import seaborn as sns


# In[25]:


sns.lmplot(x="Item_Weight", y="Item_MRP", data=file,
           hue="Item_Outlet_Sales", fit_reg=False, legend=True)


# In[26]:


file.isnull().sum()


# In[27]:


file['Outlet_Location_Type'].unique()


# In[28]:


sns.countplot(file.Item_Fat_Content)


# In[29]:


sns.countplot(file.Item_Type)
plt.xticks(rotation=90)


# In[30]:


sns.countplot(file.Outlet_Type)
plt.xticks(rotation=90)


# In[31]:


file.plot(kind='scatter',x='Item_Visibility',y='Item_Outlet_Sales')


# In[32]:


file.plot(kind='bar',x='Outlet_Identifier',y='Item_Outlet_Sales',figsize=(20,10))


# In[ ]:





# In[35]:


youngest_store=max(file['Outlet_Establishment_Year'])


# In[36]:


youngest_store


# In[37]:


outlet_age=list(file['Outlet_Establishment_Year'])


# In[38]:


outlet_age


# In[39]:


age=[]


# In[40]:


def find_age():
    for i in outlet_age:
        x=2020-i
        age.append(x)


# In[41]:


find_age()


# In[42]:


age


# In[43]:


len(age)


# <h2>Add Age column to file</h2>

# In[44]:


file["Age"]=age


# In[45]:


file


# In[46]:


file.drop('Outlet_Establishment_Year',axis=1,inplace=True)


# In[47]:


file


# In[48]:


file.columns


# In[49]:


file.plot(kind='scatter',x='Age',y='Item_Outlet_Sales')


# <h2>Dealing with missing Values</h2>

# In[50]:


file['Item_Weight'].isnull().sum()


# In[51]:


item_weight_mean=file['Item_Weight'].mean()
item_weight_mean


# In[52]:


file['Item_Weight'].fillna(item_weight_mean, inplace = True)


# In[53]:


file['Item_Weight'].isnull().sum()


# In[54]:


file.isnull().sum()


# In[55]:


file['Outlet_Type'].unique()


# In[56]:


file['Outlet_Size'].unique()


# In[57]:


file['Outlet_Size'].isnull().sum()


# In[58]:


Outlet_size_mode=file['Outlet_Size'].mode()
Outlet_size_mode =str(Outlet_size_mode[0])


# In[59]:


file['Outlet_Size'].fillna(Outlet_size_mode, inplace=True)


# In[60]:


file['Outlet_Size'].isnull().sum()


# In[ ]:





# <h2>Feature Engineering</h2>

# In[61]:


file['Item_Fat_Content'].unique()


# In[62]:


file.dtypes['Item_Fat_Content']


# In[63]:


file['Outlet_Type'].value_counts()


# In[64]:


categorical_columns = [x for x in file.dtypes.index if file.dtypes[x]=='object']
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
for col in categorical_columns:
    print("Frequency of Categories for varible %s:" %col)
    print(file[col].value_counts())


# <h2>Combining Item_fat_category</h2>

# In[65]:


lis_lf=["Low Fat","LF","low fat"]
lis_reg=['Regular','reg']

file['Item_Fat_Content']=file['Item_Fat_Content'].replace(["Low Fat","LF","low fat"],"Low Fat")
file['Item_Fat_Content']=file['Item_Fat_Content'].replace(['Regular','reg'],"Regular")


# In[66]:


file['Item_Fat_Content'].value_counts()


# In[67]:


file.describe()


# <h2>Replacing Zeroes in Item Visibility</h2>

# In[68]:


def num_of_zeroes():
    count=0
    for i in file['Item_Visibility']:
        if i==0:
            count+=1
    return count


# In[69]:


num_of_zeroes()


# In[70]:


file['Item_Visibility'].head(10)


# In[80]:


item_visb_mean=file['Item_Visibility'].mean()
item_visb_mean


# In[81]:


file['Item_Visibility']=file['Item_Visibility'].replace(0,item_visb_mean)
file['Item_Visibility']


# In[82]:


num_of_zeroes()


# <h2>One Hot Encoding</h2>

# In[87]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
file['Outlet'] = le.fit_transform(file['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    file[i] = le.fit_transform(file[i])


# In[88]:


data = pd.get_dummies(file, columns=['Item_Fat_Content','Outlet_Location_Type',
                                     'Outlet_Size','Outlet_Type','Outlet'])


# In[89]:


data


# In[ ]:




