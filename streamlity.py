import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np


@st.cache_data
def load_data():
   
    diabetes = pd.read_csv("https://github.com/WZ1117/706_go/blob/main/data/nhanes_filtered.csv?raw=true")

    return diabetes

diabetes = load_data()

year = st.slider(
    label="Year", 
    min_value=int(diabetes["Year"].min()),
    max_value=int(diabetes["Year"].max()),
    value=2013,
    step=2
)
subset = diabetes[diabetes["Year"] == year]

user_age = st.number_input('My age is: ',min_value=10, max_value=90, value=22)
#subset = subset[subset["Sex"] == sex]

life_style_options = st.multiselect(
    'My life-style includes:',
    ["I drink more than 50 times per year.","I smoke more than 100 cigarettes times per year.","I do vigorous-intensity sports like running or basketball more than once per week"],
    ["I smoke more than 100 cigarettes times per year."])

if "I drink more than 50 times per year." in life_style_options:
    drink_yes = True
else:
    drink_yes = False
if "I smoke more than 100 cigarettes times per year." in life_style_options:
    smoke_yes = True
else:
    smoke_yes = False
if "I do vigorous-intensity sports like running or basketball more than once per week" in life_style_options:
    exercise_yes = True
else:
    exercise_yes = False

#########################################PLOT1#########################################



default_lifestyle = "Smoking"
option=["Smoking","Drinking","Exercising"]
Life_style = st.selectbox(
    label='Life-style',
    options=option,
    index=option.index(default_lifestyle)
)

smoking = subset[["Age","Glycohemoglobin_lvl","Regular_Smoker"]].dropna()
Drinking = subset[["Age","Glycohemoglobin_lvl","Regular_Drinker"]].dropna()
Exercising = subset[["Age","Glycohemoglobin_lvl","Vigorous_Exerciser"]].dropna()
list1 = list(range(18, 81))
list1 = np.array(list1)


smokingyes=smoking[smoking["Regular_Smoker"]=="yes"]
smokingno=smoking[smoking["Regular_Smoker"]=="no"]
smokingyesX = smokingyes.iloc[:, 0].values.reshape(-1, 1)  
smokingyesY =  smokingyes.iloc[:, 1].values.reshape(-1, 1)  
smokingyeslinear_regressor = LinearRegression()  
smokingyeslinear_regressor.fit(smokingyesX, smokingyesY)  
smokingyesY_pred = smokingyeslinear_regressor.predict(list1.reshape(-1, 1))  

smokingnoX = smokingno.iloc[:, 0].values.reshape(-1, 1)  
smokingnoY =  smokingno.iloc[:, 1].values.reshape(-1, 1)  
smokingnolinear_regressor = LinearRegression()  
smokingnolinear_regressor.fit(smokingnoX, smokingnoY)  
smokingnoY_pred = smokingnolinear_regressor.predict(list1.reshape(-1, 1)) 



Drinkingyes=Drinking[Drinking["Regular_Drinker"]=="yes"]
Drinkingno=Drinking[Drinking["Regular_Drinker"]=="no"]
DrinkingyesX = Drinkingyes.iloc[:, 0].values.reshape(-1, 1)  
DrinkingyesY =  Drinkingyes.iloc[:, 1].values.reshape(-1, 1)  
Drinkingyeslinear_regressor = LinearRegression()  
Drinkingyeslinear_regressor.fit(DrinkingyesX, DrinkingyesY)  
DrinkingyesY_pred = Drinkingyeslinear_regressor.predict(list1.reshape(-1, 1))  

DrinkingnoX = Drinkingno.iloc[:, 0].values.reshape(-1, 1)  
DrinkingnoY =  Drinkingno.iloc[:, 1].values.reshape(-1, 1)  
Drinkingnolinear_regressor = LinearRegression()  
Drinkingnolinear_regressor.fit(DrinkingnoX, DrinkingnoY)  
DrinkingnoY_pred = Drinkingnolinear_regressor.predict(list1.reshape(-1, 1))   


list2 = list(range(12, 81))
list2 = np.array(list2)
Exercisingyes=Exercising[Exercising["Vigorous_Exerciser"]=="yes"]
Exercisingno=Exercising[Exercising["Vigorous_Exerciser"]=="no"]
ExercisingyesX = Exercisingyes.iloc[:, 0].values.reshape(-1, 1)  
ExercisingyesY =  Exercisingyes.iloc[:, 1].values.reshape(-1, 1)  
Exercisingyeslinear_regressor = LinearRegression()  
Exercisingyeslinear_regressor.fit(ExercisingyesX, ExercisingyesY)  
ExercisingyesY_pred = Exercisingyeslinear_regressor.predict(list2.reshape(-1, 1))  

ExercisingnoX = Exercisingno.iloc[:, 0].values.reshape(-1, 1)  
ExercisingnoY =  Exercisingno.iloc[:, 1].values.reshape(-1, 1)  
Exercisingnolinear_regressor = LinearRegression()  
Exercisingnolinear_regressor.fit(ExercisingnoX, ExercisingnoY)  
ExercisingnoY_pred = Exercisingnolinear_regressor.predict(list2.reshape(-1, 1))  


if Life_style == "Exercising":
    points = alt.Chart(Exercising).mark_point().encode(
    x=alt.X('Age:Q', scale=alt.Scale(domain=[12, 80])),
    y=alt.Y('Glycohemoglobin_lvl:Q', scale=alt.Scale(domain=[2, 19]),title='Glycated hemoglobin(%)'),
    color=alt.Color("Vigorous_Exerciser", legend=alt.Legend(title=""))
).properties(
    title=str(year)+" Distribution of Blood Glucose Levels of Different Fitness Level",
    width=1100,height=600
)
    line =alt.Chart(pd.DataFrame({'ExercisingY_pred':ExercisingnoY_pred.flatten().tolist()+ExercisingyesY_pred.flatten().tolist(),'Age': list(range(12, 81))+ list(range(12, 81)), 'Group':["Non-exercise Prediction"] * 69+["Exercise Prediction"]*69})).mark_line(shape="stroke").encode(
           alt.X('Age', scale=alt.Scale(domain=[12, 80])),
           alt.Y('ExercisingY_pred',scale=alt.Scale(domain=[2, 19])),
           color=alt.Color('Group',title=""))
    if exercise_yes:
        Prediction = Exercisingyeslinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    else:
        Prediction = Exercisingnolinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    you = alt.Chart(pd.DataFrame({'Age': [user_age], 'Prediction': Prediction.flatten().tolist()})).mark_point(filled=True,shape="cross",size=120).encode(
    x=alt.X('Age:Q'),
    y=alt.Y('Prediction:Q'),
    color=alt.value('black')
)
    
elif Life_style == "Drinking":
    points = alt.Chart(Drinking).mark_point().encode(
    x=alt.X('Age:Q', scale=alt.Scale(domain=[18, 80])),
    y=alt.Y('Glycohemoglobin_lvl:Q', scale=alt.Scale(domain=[2, 19]),title='Glycated hemoglobin(%)'),
    color=alt.Color("Regular_Drinker", legend=alt.Legend(title=""))
).properties(
    title=str(year)+" Distribution of Blood Glucose Levels of Different Alcohol Consumption",
    width=1100,height=600
)
    line =alt.Chart(pd.DataFrame({'DrinkingY_pred':DrinkingnoY_pred.flatten().tolist()+DrinkingyesY_pred.flatten().tolist(),'Age': list(range(18, 81))+ list(range(18, 81)), 'Group':["Non-drink Prediction"] * 63+["Drink Prediction"]*63})).mark_line(shape="stroke").encode(
           alt.X('Age', scale=alt.Scale(domain=[18, 80])),
           alt.Y('DrinkingY_pred',scale=alt.Scale(domain=[2, 19])),
           color=alt.Color('Group', title="")
)
    if drink_yes:
        Prediction = Drinkingyeslinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    else:
        Prediction = Drinkingnolinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    you = alt.Chart(pd.DataFrame({'Age': [user_age], 'Prediction': Prediction.flatten().tolist()})).mark_point(filled=True,shape="cross",size=120).encode(
    x=alt.X('Age:Q'),
    y=alt.Y('Prediction:Q'),
    color=alt.value('black')
)

    
else:
    points = alt.Chart(smoking).mark_point().encode(
    x=alt.X('Age:Q', scale=alt.Scale(domain=[18, 80])),
    y=alt.Y('Glycohemoglobin_lvl:Q', scale=alt.Scale(domain=[2, 19]),title='Glycated hemoglobin(%)'),
    color=alt.Color("Regular_Smoker", legend=alt.Legend(title=""))
).properties(
    title=str(year) +" Distribution of Blood Glucose Levels of Different Smoking Habits",
    width=1100,height=600
)
    line =alt.Chart(pd.DataFrame({'smokingY_pred':smokingnoY_pred.flatten().tolist()+smokingyesY_pred.flatten().tolist(),'Age': list(range(18, 81))+ list(range(18, 81)), 'Group':["Non-smoke Prediction"] * 63+["Smoke Prediction"]*63})).mark_line(shape="stroke").encode(
           alt.X('Age', scale=alt.Scale(domain=[18, 80])),
           alt.Y('smokingY_pred',scale=alt.Scale(domain=[2, 19])),
           color=alt.Color('Group', title="")
)

    if smoke_yes:
        Prediction = smokingyeslinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    else:
        Prediction = smokingnolinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    you = alt.Chart(pd.DataFrame({'Age': [user_age], 'Prediction': Prediction.flatten().tolist()})).mark_point(filled=True,shape="cross",size=120).encode(
    x=alt.X('Age:Q'),
    y=alt.Y('Prediction:Q'),
    color=alt.value('black')
)
    
# Scatter Plot
points+line+you

#########################################PLOT1#########################################





#########################################PLOT2#########################################
# Create DataFrame
def sum_pairs(lst):
    result = []
    for i in range(0, len(lst)-1, 2):
        result.append(lst[i] + lst[i+1])
        result.append(lst[i] + lst[i+1])
    return result
def calculate_rates(yes,total):
    result = []
    for i in range(1, len(yes), 2):
        result.append(yes[i] / total[i] * 100)
    return result
def calculate_avergae_rates(yes,total):
    result = []
    result2 = []
    for i in range(1, 8, 2):
        result.append((yes[i]+ yes[i+8])/ (total[i]+total[i+8]) * 100)
        result2.append((yes[i]+ yes[i+8])/ (total[i]+total[i+8]) * 100)
    for i in result:
        result2.append(i)
    return result2

total=sum_pairs(subset.groupby(["Gender","BMI_Group"]).Diabetes.value_counts().reset_index(level=0).Diabetes.to_list())
rates=calculate_rates(subset.groupby(["Gender","BMI_Group"]).Diabetes.value_counts().reset_index(level=0).Diabetes.to_list(),total)
average=calculate_avergae_rates(subset.groupby(["Gender","BMI_Group"]).Diabetes.value_counts().reset_index(level=0).Diabetes.to_list(),total)

plot2df = {'BMI_Group': ['Normal', 'Obese', 'Overweight', 'Underweight','Normal', 'Obese', 'Overweight', 'Underweight'],
        'Gender': ['Female','Female','Female','Female',"Male","Male","Male","Male"],
       "Rate":rates,
          "Average":average}
  
plot2df = pd.DataFrame(plot2df)
plot2df['Rate'] = plot2df['Rate'].apply(lambda x: float("{:.2f}".format(x)))
plot2df['Average'] = plot2df['Average'].apply(lambda x: float("{:.2f}".format(x)))


# Create Plot
plot2 = alt.Chart(plot2df).mark_bar().encode(alt.X('Gender',axis=None),alt.Y('Rate', title='Rates(%)'),color=alt.Color('Gender'),column=alt.Column('BMI_Group', sort=[ 'Underweight', 'Normal', 'Overweight','Obese'], header=alt.Header(titleFontSize=10)),tooltip= [alt.Tooltip(field = "Rate",title = "Rate(%)"), alt.Tooltip(field = "Average", title = "Average Rate(%)")]).properties(
    title=str(year)+' Diabetes Rates Vs. BMI').properties(
    width=alt.Step(65) 
).configure_axis(
    labelFontSize=20)
user_BMI="Normal"
line2 =alt.Chart(plot2df).mark_line(shape="stroke").encode(
           alt.X('BMI_Group', sort=[ 'Underweight', 'Normal', 'Overweight','Obese']),
           alt.Y('Average'),
    shape=alt.condition(alt.datum.BMI_Group == user_BMI,alt.value('cross'),     # which sets the bar orange.
        alt.value('circle')),
color=alt.condition(alt.datum.BMI_Group == user_BMI,alt.value('red'),     # which sets the bar orange.
        alt.value('blue'))).properties(
     title=str(year)+' Overall Diabetes Rates of Different BMI Group',
    width=650 
)

plot2
line2
#########################################PLOT2#########################################