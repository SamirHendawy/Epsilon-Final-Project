
## Import Libraries
import numpy as np
import pandas as pd
import os
import plotly.express as px
import streamlit as st
import joblib
from utils import new_process

## Load the model with joblib
model = joblib.load("xgboost.pkl")

## Define the path to the train.csv file
TRAIN_PATH = os.path.join(os.getcwd(), "train.csv")
## Read the train.csv file into a DataFrame, using "Unnamed: 0" column as the index
df_train = pd.read_csv(TRAIN_PATH, index_col="Unnamed: 0")
## Define the path to the test.csv file
TEST_PATH = os.path.join(os.getcwd(), "test.csv")
## Read the test.csv file into a DataFrame, using "Unnamed: 0" column as the index
df_test = pd.read_csv(TEST_PATH, index_col="Unnamed: 0")
## Concatenate the train and test DataFrames vertically to create a combined DataFrame
df = pd.concat([df_train, df_test], axis=0)

df.drop(columns=["id"], inplace=True)
convert_2_str = ['Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink',
                 'Online boarding', 'Seat comfort','Inflight entertainment', 'On-board service', 'Leg room service','Baggage handling', 'Checkin service', 'Inflight service','Cleanliness']
df[convert_2_str] = df[convert_2_str].astype("str")

st.set_page_config(
    layout='wide',
    page_title='Airline Passenger Satisfaction',
    page_icon='✈️'
)

st.write("<h1 style='text-align: center; color: green'> ✈ Airline Passenger Satisfaction.</h1>", unsafe_allow_html=True)
st.image("airline.jpg", use_column_width=True)

tab1, tab2, tab3, tab4 = st.tabs(['DataSet Overview', 'Conclusions', 'Describtive Statistics', "Deployment"])

with tab1:
    
    link_text = "Dataset from Kaggle website"
    link_url = "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data"
    # Create a hyperlink using st.markdown
    st.markdown(f"[{link_text}]({link_url})")
    
    st.dataframe(df.head(10))
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.write("<h4 style='color: green'>The Purpose of Project</h4>", unsafe_allow_html=True)
    st.write("<b>What factors are highly correlated to a satisfied (or dissatisfied) passenger? Can you predict passenger satisfaction?</b>", unsafe_allow_html=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.write("<h4 style='color: green'>Dataset Attributes</h4>", unsafe_allow_html=True)
    st.write("""* <code>Gender</code> - Gender of the passengers (Female, Male)
* <code>Customer Type</code> - The customer type (Loyal customer, disloyal customer)
* <code>Age</code> - The actual age of the passengers
* <code>Type of Travel</code> - Purpose of the flight of the passengers (Personal Travel, Business Travel)
* <code>Class</code> - Travel class in the plane of the passengers (Business, Eco, Eco Plus)
* <code>Flight distance</code> - The flight distance of this journey
* <code>Inflight wifi service</code> - Satisfaction level of the inflight wifi service 
* <code>Departure/Arrival time convenient</code> - Satisfaction level of Departure/Arrival time convenient
* <code>Ease of Online booking</code> - Satisfaction level of online booking
* <code>Gate location</code> - Satisfaction level of Gate location
* <code>Food and drink</code> - Satisfaction level of Food and drink
* <code>Online boarding</code> - Satisfaction level of online boarding
* <code>Seat comfort</code> - Satisfaction level of Seat comfort
* <code>Inflight entertainment</code> - Satisfaction level of inflight entertainment
* <code>On-board service</code> - Satisfaction level of On-board service
* <code>Leg room service</code> - Satisfaction level of Leg room service
* <code>Baggage handling</code> - Satisfaction level of baggage handling
* <code>Check-in service</code> - Satisfaction level of Check-in service
* <code>Inflight service</code> - Satisfaction level of inflight service
* <code>Cleanliness</code> - Satisfaction level of Cleanliness
* <code>Departure Delay in Minutes</code> - Minutes delayed when departure
* <code>Arrival Delay in Minutes</code> - Minutes delayed when Arrival
* <code>Satisfaction</code> - Airline satisfaction level(Satisfaction, neutral or dissatisfaction)""", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

with tab2:
    
    st.write("<h3 style='color: green'>Exploratory Data Analysis (EDA).</h3>", unsafe_allow_html=True)
    
    st.write("<b>- The data contains 129,880 rows and 24 columns.</b>", unsafe_allow_html=True)
    st.write("<b>- There is a missing value in Arrival Delay in Minutes column of about 0.3%.</b>", unsafe_allow_html=True)
    st.write("<b>- There are no duplicate rows in the data</b>", unsafe_allow_html=True)
    st.write("<b>- The distribution of females and males is almost similar in the data.</b>", unsafe_allow_html=True)
    st.write("<b>- 55% of the passengers in satisfaction Column are dissatisfied. (imbalanced class) </b>", unsafe_allow_html=True)
    st.write("<b>- The flight distance contains outliers values.</b>", unsafe_allow_html=True)
    show_gph = st.checkbox('Show flight distance distribution', False)
    if show_gph:
        st.plotly_chart(px.histogram(df["Flight Distance"], marginal="box", color_discrete_sequence=px.colors.sequential.Greens_r))
    st.write("<b>- 81% of the passengers are Loyal Customers.</b>", unsafe_allow_html=True)
    st.write("<b>- Wi-Fi on board seems good.</b>", unsafe_allow_html=True)
    st.write("<b>- The majority of passengers voted that the seats were comfortable.</b>", unsafe_allow_html=True)
    st.write("<b>- The majority of travelers expressed their opinion that the arrival and departure time was appropriate.</b>", unsafe_allow_html=True)
    st.write("<b>- Most passengers choose Business and Economy class.</b>", unsafe_allow_html=True)
    show_gph = st.checkbox('Show Pie chart for Class Column', False)
    if show_gph:
        st.plotly_chart(px.pie(data_frame=df, names="Class", color_discrete_sequence=px.colors.sequential.Greens_r))
    st.write("<b>- Most passengers are between the ages of 20 and 60. </b>", unsafe_allow_html=True)
    st.write("<b>- business travel ( 69% ) is more common than personal travel ( 31% ).</b>", unsafe_allow_html=True)
    show_gph = st.checkbox('Type of Travel distribution', False)
    if show_gph:
        st.plotly_chart(px.histogram(df["Type of Travel"], color_discrete_sequence=px.colors.sequential.Greens_r))
    st.write("<h5 style='color: red'>⚠ The satisfaction level with the services is relatively average, and efforts should be made to improve the quality of services in order to meet passenger expectations.</h5>", unsafe_allow_html=True)
    st.write("<b>- In Personal Travel, passengers who are neutral or dissatisfied far outnumber the satisfied ones.</b>", unsafe_allow_html=True)
    st.write("<b>- In business travel, the proportion of satisfied passengers is higher at 58%.</b>", unsafe_allow_html=True)
    st.write("<b>- There is no relationship between flight distance and arrival delay.</b>", unsafe_allow_html=True)
    st.write("<b>- In business travel, the proportion of satisfied passengers is higher at 58%.</b>", unsafe_allow_html=True)
    st.write("<b>- There is a weak Correlation between cleanliness and overall passenger satisfaction.</b>", unsafe_allow_html=True)
    st.write("<b>- A large proportion of Business passengers have a high satisfaction level.</b>", unsafe_allow_html=True)
    st.write("<b>- There are outliers values in the business category.</b>", unsafe_allow_html=True)
    st.write("<b>- The correlation between Arrival Delay and Departure Delay is very strong.</b>", unsafe_allow_html=True)
    st.write("<b>- In personal travel, the average satisfaction rate with food and drink is equal in both cases (satisfied , dissatisfied).</b>", unsafe_allow_html=True)
    st.write("<b>- In business travel, the average rate of satisfaction with food and drink is higher in satisfied case than in dissatisfied case.</b>", unsafe_allow_html=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.write("<h3 style='color: green'>Models.</h3>", unsafe_allow_html=True)
    
    st.write('<h5>I used the following algorithms in the project:</h5>', unsafe_allow_html=True)
    st.write("<b>- Logistic Regression.</b>", unsafe_allow_html=True)
    st.write("<b>- K-Nearest Neighbors Algorithm.</b>", unsafe_allow_html=True)
    st.write("<b>- Naïve Bayes Classifier.</b>", unsafe_allow_html=True)
    st.write("<b>- Support Vector Machine Using Poly and RBF Kernals.</b>", unsafe_allow_html=True)
    st.write("<b>- Random Forest.</b>", unsafe_allow_html=True)
    st.write("<b>- Voting Classifier.</b>", unsafe_allow_html=True)
    st.write("<b>- XGBoost.</b>", unsafe_allow_html=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    st.write("<h2 style='text-align: center; color: green'>I will use XGBoost in deployment because it achieved the highest performance of 95.6% (f1-score).</h2>", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

with tab3:
    st.write("<h3 style='text-align: center; color: green'>Descriptive Statistics.</h3>", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    column1, space, column2 = st.columns([5,1,5])
    with column1:
        st.metric("Mean Age of Passengers", round(df["Age"].mean()))
        st.metric("Mean Flight Distance", round(df["Flight Distance"].mean()))
        st.dataframe(df[["Flight Distance", "Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]].describe(), width=540, height=315)
    with column2:
        st.metric("The Mode Value in the Travel of Type", df["Type of Travel"].mode()[0])
        st.metric("The Mode Value in the Class", df["Class"].mode()[0])
        st.metric("The Mode Value in the Customer Type", df["Customer Type"].mode()[0])
        st.dataframe(df.describe(exclude="number"), width=550, height=180)
    st.markdown('<hr>', unsafe_allow_html=True)
    
with tab4:
    st.write("<h4 style='text-align: center; color: green'> Deployment.</h4>", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    
    def satisfaction_deployment():
        
        ## categ.
        Customer_Type = st.selectbox("Select Customer Type", options=['Loyal Customer', 'disloyal Customer'])
        type_of_travel = st.selectbox("Select Type of Travel", options=['Personal Travel', 'Business travel'])
        Class = st.selectbox("Select Class", options=['Eco Plus', 'Business', 'Eco'])

        ## nums.
        flight_Distance = st.number_input("input flight distance".title(), value=int(df["Flight Distance"].mean()), step=100)
        arrival_delay = st.number_input("input Arrival Delay (Minutes)".title(), value=df["Arrival Delay in Minutes"].median(), step=10.0)
        age = st.slider("Age", min_value=1, max_value=100, step=1)
        
        ## ready.
        Leg_room = st.radio("Select Leg room service rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        food_drink = st.radio("Select Food and drink rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        online_boarding = st.radio("Select Online boarding rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        baggage_handling = st.radio("Select Baggage handling rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        onboard_service = st.radio("Select On-board service rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        inflight_wifi = st.radio("Select Inflight wifi service rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        online_booking = st.radio("Select Online booking rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        cleanliness = st.radio("Select Cleanliness rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        inflight_entertainment = st.radio("Select Inflight entertainment rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        inflight_service = st.radio("Select Inflight service rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        seat_comfort = st.radio("Select Seat comfort rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)
        checkin_service = st.radio("Select Checkin service rate", options=[0, 1, 2, 3, 4, 5] , horizontal=True)        
        
        ## Create a button
        if st.button('Predict Satsfication', help='Click to predict satisfaction'):
            new_input = np.array([flight_Distance, arrival_delay, age,
                                  Customer_Type, type_of_travel, Class,
                                  online_booking, Leg_room, online_boarding, inflight_service,
                                  inflight_wifi, food_drink, inflight_entertainment, cleanliness,
                                  onboard_service, baggage_handling, seat_comfort, checkin_service])
            
            sample_processed = new_process(new_sample=new_input)
            pred = model.predict(sample_processed)
            
            ## "dissatisfied" ==> 0
            ## "satisfied" ==> 1
            if pred[0] == 1:
                pred = "satisfied"
            elif pred[0] == 0:
                pred = "dissatisfied"
            
            ## Display Results
            st.success(f'Satsfication Prediction is : {(pred.title())}')
            
            if (pred == "satisfied"):
                st.image("satisfied.jpg")
            elif (pred =="dissatisfied"):
                st.image("dissatisfied.jpg")
            
    if __name__ == '__main__':
        satisfaction_deployment() ## call "satisfaction_deployment" function..
        
