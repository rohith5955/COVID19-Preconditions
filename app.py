import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
from PIL import Image

@st.cache(allow_output_mutation=True, suppress_st_warning=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
               
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

st.markdown('<style>h1{color: maroon;}</style>', unsafe_allow_html=True)
st.markdown('<style>h2{color: darkblue;}</style>', unsafe_allow_html=True)

def covidrisk(row, model, feat_cols):
    df = pd.DataFrame([row], columns = feat_cols)
    features = pd.DataFrame(df, columns = feat_cols)
    if (model.predict(features)==0):
        return "LOW risk of contracting COVID-19"
    elif (model.predict(features)==1):
        return "HIGH risk of contracting COVID-19"

st.title('COVID-19 Pre-conditions Prediction')
image = Image.open('Data/covid.png')
st.image(image, width=500)

st.markdown('The code, dataset and models used are available in the GitHub repository at https://github.com/rohith5955/COVID19-Preconditions')
st.markdown('The left sidebar allows you to simulate the risk of contracting COVID-19 based on demographics, and certain pre-existing conditions, if any')

gender_select = ['Male', 'Female']
tf = {False: 0, True: 1}
patient_types = ["Out-patient", "In-patient"]

# Header
st.sidebar.header('Patient data')

# Demographics Sub-header
st.sidebar.subheader('Demographics')

age = st.sidebar.number_input("Age in Years", 1, 150, 25, 1)
sex = st.sidebar.selectbox("Sex", gender_select)
gender_bool = 0
if sex == 'Female':
    gender_bool = 1
elif sex == 'Male':
    gender_bool = 0
if sex == 'Female':
    pregnancy = st.sidebar.selectbox("Pregnancy", tf)
else:
    pregnancy = 0
patient_type = st.sidebar.selectbox("Patient type", patient_types)
patient_type_bool = 0
if patient_type == "Out-patient":
    patient_type_bool = 1
elif patient_type == "In-patient":
    patient_type_bool = 2

# Pre-existing conditions Sub-header
st.sidebar.subheader('Pre-existing Conditions')

# Using drop-down list
# asthma = st.sidebar.selectbox("Asthma", tf)
# pneumonia = st.sidebar.selectbox("Pneumonia", tf)
# obesity = st.sidebar.selectbox("Obesity", tf)
# diabetes = st.sidebar.selectbox('Diabetes', tf)
# hypertension = st.sidebar.selectbox("Hypertension", tf)
# tobacco = st.sidebar.selectbox("Tobacco", tf)
# cardiovascular = st.sidebar.selectbox("Cardiovascular", tf)
# renal_chronic = st.sidebar.selectbox("Renal chronic", tf)
# contact_other_covid = st.sidebar.selectbox("Contact with COVID patients?", tf)

# Using checkboxes
asthma = st.sidebar.checkbox("Asthma")
pneumonia = st.sidebar.checkbox("Pneumonia")
obesity = st.sidebar.checkbox("Obesity")
diabetes = st.sidebar.checkbox('Diabetes')
hypertension = st.sidebar.checkbox("Hypertension")
tobacco = st.sidebar.checkbox("Tobacco")
cardiovascular = st.sidebar.checkbox("Cardiovascular")
renal_chronic = st.sidebar.checkbox("Renal chronic")
contact_other_covid = st.sidebar.checkbox("Contact with COVID patients?")

# Collect all values to create a predictor set
row = [gender_bool, patient_type_bool, pneumonia, age, pregnancy, diabetes, asthma, hypertension, cardiovascular, obesity, renal_chronic, tobacco, contact_other_covid]

if (st.button('🦠 Check COVID-19 Risk')):
    feat_cols = ['sex', 'patient_type', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'asthma', 'hypertension', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid']
    model = joblib.load('models/mlp.joblib')
    result = covidrisk(row, model, feat_cols)
    
    # def highlight_survived(s):
    #     return ['background-color: green']*len(s) if s.Survived else ['background-color: red']*len(s)

    # def color_survived(val):
    #     color = 'green' if val else 'red'
    #     return f'background-color: {color}'
    if result == "HIGH risk of contracting COVID-19":
        st.markdown('<style>h2{color: red;}</style>', unsafe_allow_html=True)
    elif "LOW risk of contracting COVID-19":
        st.markdown('<style>h2{color: darkgreen;}</style>', unsafe_allow_html=True)

    st.header(result)
    
