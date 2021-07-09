#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd


# In[2]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))


# In[3]:


@app.route('/')
def home():
    r = render_template('index.html', role = 'Select Role', qualification = 'Select Qualification')
    return r

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    
    min_qualicication = int_features[0]
    role_cleaned = int_features[1]
    avg_experience_yrs = int_features[2]
    current_salary = int_features[3]
    
    print('min_qualicication :: ',min_qualicication, '\nrole_cleaned :: ',role_cleaned,
          '\navg_experience_yrs :: ',avg_experience_yrs, '\ncurrent_salary', current_salary)
    
    min_qualicication = le.transform([min_qualicication])
    role_cleaned = le.transform([role_cleaned])
    
    print('min_qualicication :: ',min_qualicication[0], '\nrole_cleaned :: ',role_cleaned[0],
          '\navg_experience_yrs :: ',avg_experience_yrs, '\ncurrent_salary', current_salary)
    
    final_features = scaler.transform([[min_qualicication[0], role_cleaned[0], avg_experience_yrs]])
    
    prediction = model.predict(final_features)
    print('prediction :: ',prediction)
    output = round(prediction[0], 2)
    
    text = ''
    if len(current_salary) > 0 :
        #Your current salary is 37.5% less than the market rate 
        c = float(output) - float(current_salary)
        print('c : ',c)
        if c < 0:
            c = round(abs(c)*100/float(output),2 ) 
            print('c ',c)
            text = f'\nYour current salary is {c}% more than the market rate'
        elif c > 0:
            c = round(abs(c)*100/float(output),2 )
            print('c ',c)
            text = f'\nYour current salary is {c}% less than the market rate'

    return render_template('index.html', prediction_text=f'Employee Salary should be ₹ {output} Lacs',
                          prediction_text2 = text, exp = int_features[2], sal = int_features[3], role = int_features[1],
                          qualification = int_features[0])


# In[ ]:


if __name__ == '__main__':
    app.run(debug=False)


# In[ ]:





# In[ ]:





# In[ ]:




