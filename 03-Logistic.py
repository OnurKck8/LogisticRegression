#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# In[5]:


data=pd.read_excel('/Users/onurkck/Desktop/PythonData/Data/GeneralLearn.xlsx')


# In[6]:


print(data.head(3))


# In[7]:


# Label Encoding işlemi
data['Region'] = data['Region'].map({'İstanbul': 1})


# In[8]:


# X ve y değişkenlerini belirleme
X = data[['Product_Prise', 'Advertising_Expenditure']]
y = data[['Sales_Quantity']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # %80 eğit  %20 tahmin et

# Logistic regresyon modeli oluşturma ve eğitme
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[23]:


# Eğitim ve test veri setleri üzerinde tahmin yapma
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Eğitim ve test veri setleri için doğruluk oranı hesaplama
accuracy_train = model.score(X_train, y_train)
accuracy_test = model.score(X_test, y_test)

print('Eğitim Seti Doğruluk Oranı:', accuracy_train)
print('Test Seti Doğruluk Oranı:', accuracy_test)


# In[16]:


#Neyi tahmin edeceksin
prediction_april_sales = model.predict([[20,120]])
print('Satılıp, Satılmayacak Tahmini: ', prediction_april_sales)


# In[17]:


# Görselleştirmeye lojistik regresyon eğrisini ekleyin
plt.scatter(data['Product_Prise'], data['Sales_Quantity'], label='Gerçek Satış')
plt.plot(data['Product_Prise'], model.predict(X), color='red', label='Tahmini Satış')
plt.xlabel('Product_Prise')
plt.ylabel('Sales_Quantity')
plt.title('Lojistik Regresyon Modeli ile Tahmini Satış')
plt.legend()
plt.show()

