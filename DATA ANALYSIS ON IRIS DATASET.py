#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


cd ..


# In[3]:


ls|grep csv


# <h2> LOADING DATA FILE</h2>

# In[4]:


file=pd.read_csv("Iris.csv")


# In[5]:


file.head(80)


# In[8]:


type(file)


# In[14]:


file.columns


# <h2>Check whether dataset is balanced or imbalanced</h2>

# In[9]:


file["Species"].value_counts() #DATASET IS BALANCED


# <h2>Applying Descriptive Function on Dataset</h2>

# In[10]:


#MEAN
lis_mean=[]
def mean(lis):
    
    for i in range(1,len(lis.columns)-1):
        sum=0
        for j in range(len(lis)):
            sum+=lis.iloc[j][i]    
        average=sum/len(lis)
        lis_mean.append(average)
    return lis_mean
mean(file)


# In[70]:


#MODE

def mode1(data):
      for i in range(1,len(data.columns)-1):
        mod={}
        for j in range(len(data)):
            res=data.iloc[j][i]
            if res in mod:
                mod[res]+=1
            else:
                mod[res]=1
            #print(mod)   
        maxmod=max(mod.values())
        listofkeys=[]
        for key, value in mod.items():
            if value == maxmod:
                listofkeys.append(key)
        print(listofkeys)
mode1(file)            


# In[110]:


file.iloc[1][0]


# In[18]:


file.describe()


# <h2>Looking For the Important Feature</h2>

# In[20]:


file.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm')


# In[65]:


import seaborn as sns
sns.lmplot(x="SepalLengthCm", y="SepalWidthCm", data=file,
           hue="Species", fit_reg=False, legend=True)


# In[67]:


sns.lmplot(x="SepalLengthCm", y="PetalLengthCm", data=file,
           hue="Species", fit_reg=False, legend=True)


# In[68]:


sns.lmplot(x="SepalLengthCm", y="PetalWidthCm", data=file,
           hue="Species", fit_reg=False, legend=True)


# In[69]:


sns.lmplot(x="SepalLengthCm", y="PetalLengthCm", data=file,
           hue="Species", fit_reg=False, legend=True)


# In[70]:


sns.lmplot(x="SepalLengthCm", y="PetalWidthCm", data=file,
           hue="Species", fit_reg=False, legend=True)


# In[71]:


sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", data=file,
           hue="Species", fit_reg=False, legend=True)


# In[37]:


#sns.pairplot(file, hue="Species", height=3)


# <h3>Looking for Important Feature Using Univariate Analysis </h3>

# In[139]:


x1 = file.loc[file.Species=='Iris-setosa','PetalLengthCm']
x2 = file.loc[file.Species=='Iris-versicolor','PetalLengthCm']
x3 = file.loc[file.Species=='Iris-virginica','PetalLengthCm']
plt.hist(x1, color='g', label='Iris-setosa')
plt.hist(x2,  color='b', label='Iris-versicolor')
plt.hist(x3, color='r', label='Iris-virginica')
plt.legend();
plt.xlabel('PetalLengthCm')


# In[14]:


plt.scatter(file['PetalLengthCm'],file['Species'])


# In[15]:


x1 = file.loc[file.Species=='Iris-setosa','PetalWidthCm']
x2 = file.loc[file.Species=='Iris-versicolor','PetalWidthCm']
x3 = file.loc[file.Species=='Iris-virginica','PetalWidthCm']
plt.hist(x1, color='g', label='Iris-setosa')
plt.hist(x2,  color='b', label='Iris-versicolor')
plt.hist(x3, color='r', label='Iris-virginica')
plt.legend();
plt.xlabel('PetalWidthCm')


# In[16]:


x1 = file.loc[file.Species=='Iris-setosa','SepalLengthCm']
x2 = file.loc[file.Species=='Iris-versicolor','SepalLengthCm']
x3 = file.loc[file.Species=='Iris-virginica','SepalLengthCm']
plt.hist(x1, color='g', label='Iris-setosa')
plt.hist(x2,  color='b', label='Iris-versicolor')
plt.hist(x3, color='r', label='Iris-virginica')
plt.legend();
plt.xlabel('SepalLengthCm')


# In[69]:


x1 = file.loc[file.Species=='Iris-setosa','SepalWidthCm']
x2 = file.loc[file.Species=='Iris-versicolor','SepalWidthCm']
x3 = file.loc[file.Species=='Iris-virginica','SepalWidthCm']
plt.hist(x1, color='g', label='Iris-setosa')
plt.hist(x2,  color='b', label='Iris-versicolor')
plt.hist(x3, color='r', label='Iris-virginica')
plt.legend();
plt.xlabel('SepalWidthCm')


# In[49]:


df=file[file.columns[1:]]
corr = df.corr(method="pearson")
corr


# In[67]:


sns.boxplot(x='Species',y='PetalLengthCm',data=file)


# In[19]:


from scipy.stats import norm


# In[31]:


file.PetalLengthCm.skew()


# In[32]:


file.PetalLengthCm.kurt()


# In[36]:


file['Species'].unique()


# In[47]:


iris_setosa = file.loc[file["Species"] == "Iris-setosa"]
iris_virginica = file.loc[file["Species"] == "Iris-virginica"]
iris_versicolor = file.loc[file["Species"] == "Iris-versicolor"]
counts, bin_edges = np.histogram(iris_setosa['PetalLengthCm'], bins=10)
pdf = counts/(sum(counts))
print(pdf) 
print(bin_edges)
cdf = np.cumsum(pdf)
plt.grid()
plt.plot(bin_edges[1:],pdf,label="PDF")
plt.plot(bin_edges[1:], cdf,label="CDF")
plt.legend()


# In[ ]:




