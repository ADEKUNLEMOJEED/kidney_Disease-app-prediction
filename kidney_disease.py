import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


data = pd.read_csv('kidney_disease.csv')

data.rename(columns={'bp': 'blood_pressure', 'sg': 'specific gravity','al':'albumin','su':'sugar','rbc':'red blood cells',
                     'pc':'pus cell','pcc':'pus cell clumps','ba':'bacteria','bgr':'blood glucose random','bu':'blood urea',
                     'sc':'serum creatinine','sod':'sodium','pot':'potassium','hemo':'hemoglobin','pcv':'packed cell volume',
                     'wc':'white blood cell count','rc':'red blood cell count','htn':'hypertension','dm':'diabetes mellitus',
                     'cad':'coronary artery disease','appet':'appetite','pe':'pedal edema','ane':'anemia','classification':'classes'},inplace=True)

# We want to remove the 'ckd\t' class becuase of its imbalance
data = data.loc[data['classes'] != 'ckd\t']
data['classes'].value_counts()

df = data.copy()

def converter(dataframe, column_name, new_name):
    dataframe[new_name] = dataframe[column_name].str.extract(r'(\d+)') 
    dataframe[new_name] = dataframe[new_name].astype(float) 
    dataframe.drop(column_name, axis=1, inplace=True)   

converter(df, 'packed cell volume', 'packed_cell_volume')
converter(df, 'white blood cell count', 'white_blood_cell_count')
converter(df, 'red blood cell count', 'red_blood_cell_count')

# Select your strategy to handle missing values, and tell us why you had made that choice.
for i in df.columns: # ------------------------------------------------------- Iterate through the ds columns
    if ((df[i].isnull().sum()/ len(df)) * 100) > 30: # --------------------- If the missing value of a column is above 30%
        df.drop([i], axis = 1, inplace = True) # ----------------------------- Drop the column

    elif df[i].dtypes == 'O': # ---------------------------------------------- Or if its not above 30%, and its categorical
        df[i].fillna(df[i].mode()[0], inplace = True) # -------------------- fill the missing value with the mode

    else:
        df[i].fillna(df[i].median(), inplace = True) # --------------------- We are left with only numerical ds, fill the missing with median

df.isnull().sum()

sel = df.copy()

categoricals = df.select_dtypes(include = ['object', 'category'])
numericals = df.select_dtypes(include = 'number')

# PREPROCESSSING
# Standardization
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
encoder = LabelEncoder()

for i in numericals.columns: # ................................................. Select all numerical columns
    if i in df.columns: # ...................................................... If the selected column is found in the general dataframe
        df[i] = scaler.fit_transform(df[[i]]) # ................................ Scale it

for i in categoricals.columns: # ............................................... Select all categorical columns
    if i in df.columns: # ...................................................... If the selected columns are found in the general dataframe
        df[i] = encoder.fit_transform(df[i])# .................................. encode it

df.drop('id', axis = 1, inplace = True)
sel_cols = ['serum creatinine', 'hemoglobin', 'age', 'blood urea', 'hypertension']
dx = df[sel_cols]

x = dx
y = df.classes

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 9, stratify = y)

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier()
model.fit(xtrain, ytrain)
train_prediction = model.predict(xtrain)
print(classification_report(train_prediction, ytrain))

# save model
model = pickle.dump(model, open('kidney_Disease.pkl', 'wb'))

# # ..............STREAMLIT DEVELOPEMENT..........
# Load the model
kidney_model = pickle.load(open('kidney_Disease.pkl', 'rb'))

# Streamlit app header
st.markdown("<h1 style='color: #1F1717; text-align: center; font-family: Arial, Helvetica, sans-serif;'>Kidney Disease</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='margin: -16px; color: #D36B00; text-align: center; font-family: Arial, Helvetica, sans-serif;'>Built By Adekunle Mojeed</h5>", unsafe_allow_html=True)
st.sidebar.markdown('<br><br><br>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.image('pngwing.com (26).png', width=300)
selected_page = st.sidebar.radio("Navigation", ["Home", "Prediction"])

# Function to seline the home page content
def home_page():
    st.image('pngwing.com (22).png', width=200)
    st.markdown("<h2 style='color: #990000;'>Project Background</h2>", unsafe_allow_html=True)
    st.write("""Health Indicators:
    - **Serum Creatinine:**
      - *Description:* Measures kidney function; elevated levels may indicate impaired kidney function.
    - **Hemoglobin:**
      - *Description:* Protein in red blood cells carrying oxygen; helps assess oxygen-carrying capacity.
    - **Age:**
      - *Description:* Fundamental factor in health assessments; influences risk for various health conditions.
    - **Hypertension:**
      - *Description:* High blood pressure; a key risk factor for cardiovascular issues.
    - **Blood Urea:**
      - *Description:* Measures urea nitrogen in blood; elevated levels may signal kidney dysfunction.
    - **These health indicators provide valuable insights into different aspects of your well-being.
    Understanding and monitoring these factors can contribute to proactive health management.""")
    st.sidebar.markdown('<br>', unsafe_allow_html= True)

def prediction_page():
    st.markdown("<div style='text-align: center;'><img src='pngwing.com (23).png' alt='Kidney Disease' width='300'></div>", unsafe_allow_html=True)

    st.sidebar.markdown("Add your modeling content here")
    # st.write(sel.head())
    st.sidebar.image('pngwing.com (23).png', width=300)

    # Collect user input
    serum_creatinine = st.sidebar.slider("Serum Creatinine", sel['serum creatinine'].min(), sel['serum creatinine'].max())
    hemoglobin = st.sidebar.slider("Hemoglobin", sel['hemoglobin'].min(), sel['hemoglobin'].max())
    age = st.sidebar.slider("Age", sel['age'].min(), sel['age'].max())
    blood_urea = st.sidebar.slider("Blood Urea", sel['blood urea'].min(), sel['blood urea'].max())
    hypertension = st.sidebar.selectbox("Hypertension", sel['hypertension'])

    user_input = {
        'Serum Creatinine': serum_creatinine,
        'Hemoglobin': hemoglobin,
        'Age': age,
        'Blood Urea': blood_urea,
        'Hypertension': hypertension
    }

# Create a DataFrame from the dictionary
    input_df = pd.DataFrame(user_input, index=[0])

    st.markdown("<h4 style='text-align: left; color: #D36B00;'>USER INPUT</h4>", unsafe_allow_html=True)
    # Display the input DataFrame
    st.write(input_df)

# Preprocess the input data
    categoricals = input_df.select_dtypes(include = ['object', 'category'])
    numericals = input_df.select_dtypes(include = 'number')
        
# Standard Scale the Input Variable.
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    encoder = LabelEncoder()

    for i in numericals.columns:
        if i in input_df.columns:
            input_df[i.lower()] = scaler.fit_transform(input_df[[i]])
        input_df.drop(i, axis=1, inplace=True)
    for i in categoricals.columns:
        if i in input_df.columns:
             input_df[i.lower()] = encoder.fit_transform(input_df[i])
        input_df.drop(i, axis=1, inplace=True)

    if st.button("Predict Kidney Disease"):
        # Make prediction
        prediction =kidney_model.predict(input_df)

#       Display prediction
        st.success(f"The patient is {'Positive for Kidney Disease' if prediction[0] == 1 else 'Negative for Kidney Disease'}.")
        st.image('pngwing.com (27).png', width=300)
    
# Display content based on the dfected page
if selected_page == "Home":
    home_page()
elif selected_page == "Prediction":
    prediction_page()

