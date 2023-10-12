#!/usr/bin/env python
# coding: utf-8

# # Bank Customer Churn Prediction

# In[1]:


# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# importing the dataset
df=pd.read_csv("C:\\Users\\91789\\OneDrive\\Desktop\\customer_churn_data.csv")
df


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.describe(include=object)


# In[10]:


df.isnull().sum()


# In[11]:


df.isnull()


# In[12]:


sns.heatmap(df.isnull(), cmap='viridis')


# In[13]:


sns.heatmap(df.isnull(), cmap='Reds')


# In[14]:


df.duplicated()


# In[15]:


df[df.duplicated()]


# In[16]:


df['customer_id'].duplicatedicated()


# In[17]:


df[df['customer_id'].duplicated()]


# In[18]:


df.query("customer_id == 530490")


# In[19]:


df.gender.unique()


# In[20]:


df['gender'].value_counts()


# In[21]:


df['gender'].value_counts().sum()


# In[22]:


df['gender'].notnull().sum()


# In[23]:


df['gender'].isnull().sum()


# In[24]:


df['gender'].value_counts(dropna=False)


# In[25]:


df['gender'].value_counts(dropna=False).sum()


# In[26]:


df['gender'].value_counts(dropna=False) / len(df) * 100


# In[27]:


df['gender'].value_counts(dropna=False, normalize=True) * 100


# In[28]:


sns.countplot(x='gender', data=df)


# In[29]:


plt.pie(df['gender'].value_counts(), labels=['Male', 'Female'])
plt.show()


# In[30]:


plt.pie(df['gender'].value_counts(dropna=False), 
        labels=df['gender'].value_counts(dropna=False).index, autopct='%.2f%%')
plt.show()


# In[31]:


help(df['gender'].value_counts)


# In[32]:


df['gender'].value_counts(dropna=False).index


# In[33]:


df.head()


# In[34]:


sns.displot(df['age'])


# In[35]:


sns.kdeplot(df['age'])


# In[36]:


sns.kdeplot(df['age'], shade=True)


# In[37]:


sns.kdeplot(df['age'], fill=True)


# In[38]:


sns.histplot(df['age'])


# In[39]:


sns.displot(df['age'], kde=True)


# In[40]:


df.head()


# In[41]:


df.nunique()


# In[42]:


sns.kdeplot(df.no_of_days_subscribed)


# In[43]:


sns.kdeplot(df.maximum_daily_mins)


# In[44]:


sns.kdeplot(x=df.maximum_daily_mins, y=df.minimum_daily_mins)


# In[45]:


sns.kdeplot(df.maximum_daily_mins, hue=df.gender)


# In[46]:


df.head()


# In[47]:


sns.boxplot(y=df.no_of_days_subscribed)


# In[48]:


sns.boxplot(y=df.no_of_days_subscribed, hue=df.gender)


# In[49]:


sns.boxplot(y=df.no_of_days_subscribed, hue=df['gender'])


# In[50]:


sns.boxplot(y=df.no_of_days_subscribed, hue=df['churn'])


# In[51]:


sns.boxplot(y=df.age, hue=df['churn'])


# In[52]:


sns.boxplot(x=df['churn'], y=df.no_of_days_subscribed, data=df)


# In[53]:


sns.boxplot(x=df['churn'], y=df.no_of_days_subscribed, data=df, hue='gender')


# In[54]:


sns.boxplot(y=df.videos_watched)


# In[55]:


fig, ax = plt.subplots(1, 2, figsize=(8, 6), dpi=100)
sns.boxplot(y=df.videos_watched, ax=ax[0])
sns.boxplot(y=df.no_of_days_subscribed, ax=ax[1])
plt.show()


# In[56]:


ax


# In[57]:


fig, ax = plt.subplots(1, 2, figsize=(8, 6), dpi=100)
sns.boxplot(y=df.videos_watched, ax=ax[0])
sns.boxplot(y=df.no_of_days_subscribed, ax=ax[0])
plt.show()


# In[58]:


fig, ax = plt.subplots()
sns.boxplot(y=df.videos_watched, ax=ax)
sns.boxplot(y=df.no_of_days_subscribed, ax=ax)
plt.show()


# In[59]:


df[df.duplicated()].count()


# In[60]:


df_duplicates = df.iloc[500:550]


# In[61]:


df_duplicates.head()


# In[62]:


df1 = pd.concat([df, df_duplicates])


# In[63]:


df1.shape


# In[64]:


df1[df1.duplicated()].count()


# In[65]:


df_duplicates.info()


# In[66]:


df1.drop_duplicates()


# In[67]:


df1


# In[68]:


df1.drop_duplicates(inplace=True)
df1


# In[69]:


df.isnull().sum() / len(df) * 100


# In[70]:


plt.pie(df['gender'].value_counts(dropna=False), 
        labels=df['gender'].value_counts(dropna=False).index, autopct='%.2f%%')
plt.show()


# In[71]:


df['gender'].mode()


# In[72]:


df['gender'].mode()[0]


# In[73]:


df.fillna({'gender': df['gender'].mode()[0]})


# In[74]:


df['gender'].value_counts()


# In[75]:


df['gender'].value_counts(dropna=False)


# In[76]:


df['gender'].value_counts().idxmax()


# In[77]:


df['gender'].value_counts().max()


# In[78]:


df.fillna({'gender': df['gender'].value_counts().idxmax()}, inplace=True)


# In[79]:


df.isnull().sum()


# In[80]:


sns.boxplot(y=df.videos_watched)


# In[81]:


lower_limit = df.videos_watched.quantile(0.04)
upper_limit = df.videos_watched.quantile(0.97)


# In[82]:


lower_limit


# In[83]:


upper_limit


# In[84]:


np.where(df.videos_watched < lower_limit, lower_limit, 
         np.where(df.videos_watched > upper_limit, upper_limit, df.videos_watched))


# In[85]:


lower_limit


# In[86]:


upper_limit


# In[87]:


df.videos_watched.clip(lower=lower_limit, upper=upper_limit)


# In[88]:


df1


# In[89]:


df1['videos_watched'] = np.where(df1.videos_watched < lower_limit, lower_limit, np.where(df1.


# In[90]:


videos_watched > upper_limit, upper_limit, df1.videos_watched))df1['videos_watched'] = np.where(df1.videos_watched < lower_limit, lower_limit, np.where(df1.


# In[91]:


sns.boxplot(y=df1.videos_watched)


# In[92]:


sns.boxplot(y=df.videos_watched)


# In[93]:


df2 = df.copy()


# In[94]:


id(df)


# In[95]:


id(df2)


# In[96]:


df2.videos_watched = df2.videos_watched.clip(lower=lower_limit, upper=upper_limit)


# In[97]:


sns.boxplot(y=df2.videos_watched)


# In[98]:


sns.boxplot(y=df2.no_of_days_subscribed)


# In[99]:


lower_limit = df2.no_of_days_subscribed.quantile(0.04)
upper_limit = df2.no_of_days_subscribed.quantile(0.97)
df2.no_of_days_subscribed = df2.no_of_days_subscribed.clip(lower=lower_limit, upper=upper_limit)


# In[100]:


sns.boxplot(y=df2.no_of_days_subscribed)


# In[101]:


df.videos_watched.sort_values(ascending=True)


# In[102]:


df.videos_watched = df.videos_watched.sort_values(ascending=True)


# In[103]:


lower


# In[104]:


q3 = df.videos_watched.quantile(.75)
q1 = df.videos_watched.quantile(.25)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr


# In[105]:


lower


# In[106]:


upper


# In[107]:


df.videos_watched.clip(lower=lower, upper=upper)


# In[108]:


df.videos_watched = df.videos_watched.clip(lower=lower, upper=upper)


# In[109]:


sns.boxplot(y=df.videos_watched)


# In[110]:


df.head()


# In[111]:


pd.get_dummies(df.gender)


# In[112]:


df.info()


# In[113]:


pd.get_dummies(df.multi_screen)


# In[114]:


pd.get_dummies(df.mail_subscribed)


# In[115]:


pd.get_dummies(df.mail_subscribed, prefix='mail_subscribed')


# In[116]:


pd.get_dummies(df.multi_screen, prefix='multi_screen')


# In[117]:


sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


# In[118]:


df.corr()


# In[ ]:




