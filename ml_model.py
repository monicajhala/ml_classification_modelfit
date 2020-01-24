#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd

# In[57]:


data = pd.read_csv("merdata.csv")
print(data.head())

# In[23]:


# data.columns


# In[31]:


# data.head(30)


# In[58]:


# data['Region_code'].unique()


# In[59]:


di = {'MAH': 27, 'GJ': 24, 'HP': 2, 'DEL': 7,'TN': 33,'AP': 37, 'PB':3,'KAR':29}
data['Region_code'] = data['Region_code'].map(di)


# In[37]:


dset = data[["Vendor","Difference_in_days","Quantity-deliver-percent_y","Material_x","Title_CHAR","net_price_ratio","net_price_deviation.1","net_price_ranking","material_delivery_date_ranking","min_net_price","Approval. Dev","Production Resource/Tool Saved for Insp.1","Return to vendor","Approval dev.","Insp. dev","stock dev.","Quantity-Ranking_y","URL_ranking","gst_ranking","find_gst","RG_ranking","active_block_ranking"]]


# In[65]:
print(1)

# dset.head(10)


# In[66]:


# dset["GST_CHAR"]= dset["GST_CHAR"].apply(lambda x: 1 if x == "F"else 0)
# dset["GST_CHAR"].unique()
print(1)


# In[64]:


dset = dset.fillna(0)
# dset.head(10)


# In[63]:

print(1)

final_12 = dset.groupby(['Vendor'])[['Quantity-deliver-percent_y','Difference_in_days','Title_CHAR','net_price_ratio','net_price_deviation.1','Approval. Dev','Production Resource/Tool Saved for Insp.1','Return to vendor','Approval dev.','Insp. dev','stock dev.','Quantity-Ranking_y','material_delivery_date_ranking','URL_ranking','gst_ranking','net_price_ranking','find_gst','RG_ranking','active_block_ranking']].mean()
final_12 = final_12.reset_index()

print(1)

# In[ ]:


final_12 = final_12.iloc[1:,:]

print(1)

# In[43]:


final_12["Title_CHAR"] = final_12["Title_CHAR"].apply(lambda x: 1 if float(x) > 0 else 0)


# In[44]:


# final_12.head()
print(1)


# In[46]:


final_12["rank"] = (final_12["Quantity-Ranking_y"]*5+final_12["material_delivery_date_ranking"]*5+final_12['stock dev.']*5+final_12['Insp. dev']*5+final_12['Approval dev.']*5+final_12["net_price_ranking"]*30+final_12["RG_ranking"]*20+final_12["Title_CHAR"]*5+final_12["active_block_ranking"]*5)/85


# In[47]:


final_12["rank"] = final_12["rank"].apply(lambda x: round(float(x)+0.5) if int(str(x).split(".")[-1][0]) >= 5 else round(float(x)))


# In[48]:
print(1)



fnl = final_12[['Quantity-deliver-percent_y', 'Difference_in_days','Title_CHAR','net_price_deviation.1','find_gst','Approval. Dev','Production Resource/Tool Saved for Insp.1','Return to vendor','rank']] 


# # In[50]:


# dset = dset.fillna(0)

# print(1)

# In[51]:


fnl.to_csv("weightage_final.csv")


# In[ ]:
print('svm')


# import pandas as pd
# from sklearn import svm
# fnl.to_csv("weightage_final.csv")
# fnl = pd.read_csv("weightage_final.csv")

# # importing necessary libraries 
# from sklearn import datasets 
# from sklearn.metrics import confusion_matrix 
# from sklearn.model_selection import train_test_split 







# # training a linear SVM classifier 
# from sklearn.svm import SVC 
# svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 


# # In[72]:
# print('svm2')


# print(y_train)
# svm_predictions = svm_model_linear.predict(X_test)
# len(svm_predictions)
# #len(X_test)
# # model accuracy for X_test   
# accuracy = svm_model_linear.score(X_test, y_test)
# #creating a confusion matrix 
# cm = confusion_matrix(y_test, svm_predictions)
# print(cm)
# print(accuracy)


# # # In[49]:
# print('svm3')


# X["predicted"] = pd.Series(svm_predictions)
# X["expected"] = pd.Series(y)
# X.to_csv("svm_predictionsfinal_weightage.csv",index=False)


# In[76]:


# from sklearn.metrics import classification_report
# print(classification_report(y_test, svm_predictions))


# In[73]:


import pandas as pd
fnl.to_csv("weightage_final.csv")
data = pd.read_csv("weightage_final.csv")
################################################################################################################################

######################################### S  V  M  Model  ######################################################################
################################################################################################################################
# importing necessary libraries 
# from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 


X = data.iloc[:,:-1]
y = data.iloc[:,-1]# only rank


# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)


# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 


# In[22]:
########################################################################################################################################

##################################################### Naive Bayes Model #################################################################
########################################################################################################################################

# training the model on training set 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
print('anu')
gnb.fit(X_train, y_train)


# In[21]:


# import pandas as pd
# fnl.to_csv("output.csv")
# data = pd.read_csv("output.csv")

# # importing necessary libraries 
# # from sklearn import datasets 
# from sklearn.metrics import confusion_matrix 
# from sklearn.model_selection import train_test_split 


# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]


# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=1)


# # training a linear SVM classifier 
# from sklearn.svm import SVC 
# svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 


# In[23]:


# making predictions on the testing set 
y_pred = gnb.predict(X_test) 


# In[24]:

# calculating the accuracy of the model
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)


# In[29]:


X["predicted"] = pd.Series(y_pred)
X["expected"] = pd.Series(y)
X.to_csv("naivebayes_weightages.csv",index=False)


# # In[30]:


from sklearn.metrics import  confusion_matrix
y_actu = pd.Series( y, name='Actual')
y_pred = pd.Series(y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)


# # In[28]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# # In[ ]:

# random forest 
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=400, max_depth=5,random_state=0)
clf.fit(X, y)

