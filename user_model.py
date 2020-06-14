#!/usr/bin/env python
# coding: utf-8

# In[3]:


from joblib import dump, load
import numpy as np
model = load('financial_analysis_boston.joblib') 


# In[5]:


features = np.array([[ 1.76329132, -0.4898311 ,  0.98336806, -0.27288841,  0.96749915,
       -0.07675496,  1.09697304, -1.12576063,  1.63579367,  1.50571521,
        0.81196637,  0.42050096,  1.24376169]])
model.predict(features)

