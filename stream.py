import streamlit as st
import numpy as np
import joblib

country = st.number_input(label="country")
year = st.number_input(label="year")
location_type= st.number_input(label="location_type")
cellphone_access= st.number_input(label="cellphone_access")
household_size = st.number_input(label="household_size")
age_of_respondent = st.number_input(label="age_of_respondent")
gender_of_respondent  = st.number_input(label="gender_of_respondent")
relationship_with_head = st.number_input(label="relationship_with_head")
marital_status = st.number_input(label="marital_status")
education_level = st.number_input(label="education_level")
job_type = st.number_input(label="job_type")

test_data = np.array([country, year, location_type, cellphone_access, 
                      household_size, age_of_respondent, 
                      gender_of_respondent, relationship_with_head, marital_status, 
                      education_level, job_type ]).reshape(1,-1)

model = joblib.load(r"C:\Users\ALAA\Desktop\streamlit\checkpoint2.pkl")

if st.button("predict"):
    prediction = model.predict(test_data)
    st.write("the result is: ",prediction)