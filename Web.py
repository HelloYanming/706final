import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

alt.data_transformers.disable_max_rows()


st.set_page_config(
    layout="wide",initial_sidebar_state = "auto", page_title = "NHANES - Diabetes Trend Visualizer",
)

## Read data
@st.cache_data
def load_data():
   
    diabetes = pd.read_csv("https://github.com/WZ1117/706_go/blob/main/data/nhanes_filtered.csv?raw=true")

    return diabetes

df = load_data()
diabetes = load_data()
diabetes['Year'] = diabetes['Year'].replace(to_replace = [2009, 2011, 2013, 2015, 2017], value=['2009-2010', '2011-2012', '2013-2014', '2015-2016', '2017-2018'])

with st.sidebar: 
    
    st.title("NHANES - Diabetes Trend Visualizer")
    st.subheader("Input your information for a personalized journey:")
    year =st.select_slider(
    'Select a range of year',
    options=['2009-2010', '2011-2012', '2013-2014', '2015-2016', '2017-2018'],value="2013-2014")
    
    user_age = st.number_input('My age is: ',min_value=10, max_value=90, value=22)
    
    user_height = st.number_input('My height is (cm): ',min_value=100, max_value=250, value=160)
    
    user_weight = st.number_input('My weight is (kg): ',min_value=10, max_value=300, value=65)
    
    income_type = st.radio(
    "My family income range:",
    ('<10000','10000-20000','20000-35000', '35000-55000', '55000-75000',
       '75000-100000', '>100000'))
    
    disease = st.multiselect('I have underlying health conditions:',
                  ['High Blood Pressure', 
                   'Kidney Issue', 
                   'Coronory Heart Disease', 
                   'Thyroid Issue', 
                   'Liver Issue'],
                   ['High Blood Pressure', 
                   'Kidney Issue', 
                   'Coronory Heart Disease', 
                   'Thyroid Issue', 
                   'Liver Issue'])
    
    life_style_options = st.multiselect(
    'My life-style includes:',
    ["I drink more than 50 times per year.","I smoke more than 100 cigarettes times per year.","I do vigorous-intensity sports like running or basketball more than once per week"],
    ["I smoke more than 100 cigarettes times per year."])

    
subset = diabetes[diabetes["Year"] == year]

def my_theme():
    return {
        'config': {
            'view': {
                'height': 300,
                'width': 900,
            },
            'range': {
                'category': {'scheme':'tableau10'}
            }
        }
    }

# register the custom theme under a chosen name
alt.themes.register('my_theme', my_theme)

# enable the newly registered theme
alt.themes.enable('my_theme')

#########Age group#########
if user_age<=18:
    age_type = '<=18'
if user_age>=19 and user_age<=25:
    age_type = '19-25'
if user_age>=26 and user_age<=35:
    age_type = '26-35'
if user_age>=36 and user_age<=45:
    age_type = '36-45'   
if user_age>=46 and user_age<=55:
    age_type = '46-55'   
if user_age>=56 and user_age<=65:
    age_type = '56-65'  
if user_age>65:
    age_type = '>65' 
#########################
 
#########Life Style#########
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
###############################


#########################################PLOT1#########################################
st.header("Hemoglobin A1C level across different lifestyle groups")

default_lifestyle = "Exercising"
option=["Smoking","Drinking","Exercising"]
Life_style = st.selectbox(
    label='Choose a life-style',
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




Exercising["Vigorous_Exerciser"]=Exercising["Vigorous_Exerciser"].replace(to_replace = ["no","yes"], value=["Non-vigorous exerciser","Vigorous exerciser"])
Drinking["Regular_Drinker"]=Drinking["Regular_Drinker"].replace(to_replace = ["no","yes"], value=["Non-regular drinker","Regular drinker"])
smoking["Regular_Smoker"]=smoking["Regular_Smoker"].replace(to_replace = ["no","yes"], value=["Non-regular smoker","Regular smoker"])




if Life_style == "Exercising":
    points = alt.Chart(Exercising).mark_point(filled=True,opacity=0.5).encode(
    x=alt.X('Age:Q', scale=alt.Scale(domain=[12, 80])),
    y=alt.Y('Glycohemoglobin_lvl:Q', scale=alt.Scale(domain=[2, 19]),title='Glycated hemoglobin(%)'),
    color=alt.Color("Vigorous_Exerciser", legend=alt.Legend(title="Life-style:"))
).properties(
    title=str(year)+" Distribution of Hemoglobin A1C Levels of Different Fitness Level",
    width=1000,height=500
)
    line =alt.Chart(pd.DataFrame({'ExercisingY_pred':ExercisingnoY_pred.flatten().tolist()+ExercisingyesY_pred.flatten().tolist(),'Age': list(range(12, 81))+ list(range(12, 81)), 'Group':["Non-exercising"] * 69+["Exercising"]*69})).mark_line(shape="stroke").encode(
           alt.X('Age', scale=alt.Scale(domain=[12, 80])),
           alt.Y('ExercisingY_pred',scale=alt.Scale(domain=[2, 19])),
           color=alt.Color('Group',title="Prediction",scale=alt.Scale(scheme='dark2')),
        strokeWidth=alt.value(4))
    if exercise_yes:
        Prediction = Exercisingyeslinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    else:
        Prediction = Exercisingnolinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    you = alt.Chart(pd.DataFrame({'Age': [user_age], 'Prediction': Prediction.flatten().tolist()})).mark_point(filled=True,shape="circle",size=200).encode(
    x=alt.X('Age:Q'),
    y=alt.Y('Prediction:Q'),
    color=alt.value('red')
)
    
elif Life_style == "Drinking":
    points = alt.Chart(Drinking).mark_point(filled=True,opacity=0.5).encode(
    x=alt.X('Age:Q', scale=alt.Scale(domain=[18, 80])),
    y=alt.Y('Glycohemoglobin_lvl:Q', scale=alt.Scale(domain=[2, 19]),title='Glycated hemoglobin(%)'),
    color=alt.Color("Regular_Drinker", legend=alt.Legend(title="Life-style:"))
).properties(
    title=str(year)+" Distribution of Hemoglobin A1C Levels of Different Alcohol Consumption",
    width=1000,height=500
)
    line =alt.Chart(pd.DataFrame({'DrinkingY_pred':DrinkingnoY_pred.flatten().tolist()+DrinkingyesY_pred.flatten().tolist(),'Age': list(range(18, 81))+ list(range(18, 81)), 'Group':["Non-drinking"] * 63+["Drinking"]*63})).mark_line(shape="stroke").encode(
           alt.X('Age', scale=alt.Scale(domain=[18, 80])),
           alt.Y('DrinkingY_pred',scale=alt.Scale(domain=[2, 19])),
           color=alt.Color('Group', title="Prediction:",scale=alt.Scale(scheme='dark2')),
        strokeWidth=alt.value(4)
)
    if drink_yes:
        Prediction = Drinkingyeslinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    else:
        Prediction = Drinkingnolinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    you = alt.Chart(pd.DataFrame({'Age': [user_age], 'Prediction': Prediction.flatten().tolist()})).mark_point(filled=True,shape="circle",size=200).encode(
    x=alt.X('Age:Q'),
    y=alt.Y('Prediction:Q'),
    color=alt.value('red')
)

    
else:
    points = alt.Chart(smoking).mark_point(filled=True,opacity=0.5).encode(
    x=alt.X('Age:Q', title='Age',scale=alt.Scale(domain=[18, 80])),
    y=alt.Y('Glycohemoglobin_lvl:Q', scale=alt.Scale(domain=[2, 19]),title='Glycated hemoglobin(%)'),
    color=alt.Color("Regular_Smoker", legend=alt.Legend(title="Life-style:"))
).properties(
    title=str(year) +" Distribution of Hemoglobin A1C Levels of Different Smoking Habits",
    width=1000,height=500
)
    line =alt.Chart(pd.DataFrame({'smokingY_pred':smokingnoY_pred.flatten().tolist()+smokingyesY_pred.flatten().tolist(),'Age': list(range(18, 81))+ list(range(18, 81)), 'Group':["Non-smoking"] * 63+["Smoking"]*63})).mark_line(shape="stroke").encode(
           alt.X('Age',scale=alt.Scale(domain=[18, 80])),
           alt.Y('smokingY_pred',scale=alt.Scale(domain=[2, 19])),
           color=alt.Color('Group', title="Prediction:",scale=alt.Scale(scheme='dark2')),
        strokeWidth=alt.value(4)
)

    if smoke_yes:
        Prediction = smokingyeslinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    else:
        Prediction = smokingnolinear_regressor.predict(np.array([user_age]).reshape(-1, 1))
    you = alt.Chart(pd.DataFrame({'Age': [user_age], 'Prediction': Prediction.flatten().tolist()})).mark_point(filled=True,shape="circle",size=200).encode(
    x=alt.X('Age:Q'),
    y=alt.Y('Prediction:Q'),
    color=alt.value('red')
)
    
# Scatter Plot

plt1=alt.layer(points,line,you).resolve_scale(color='independent')
plt1
st.write("Your predictive glycated hemoglobin level is highlighted in RED.")

#########################################PLOT1#########################################

#########################################PLOT3#########################################
st.header("Prevalence of diabetes based on family income & age range")
df = df.dropna(subset=["Family_Income","Age_Group","Diabetes"])
df['Diabetes'] = df['Diabetes'].map({'yes': 100, 'no': 0})


df['Year'] = df['Year'].replace(to_replace = [2009, 2011, 2013, 2015, 2017], value=['2009-2010', '2011-2012', '2013-2014', '2015-2016', '2017-2018'])
subset = df[df["Year"] == year]
subset["Age_Group"]= subset['Age_Group'].replace(to_replace = ["<18"], value=['<=18'])

# Plot of income
plot3_income = alt.Chart(subset).transform_joinaggregate(
    Rate_Family_Income='mean(Diabetes)',
    groupby=['Family_Income']
).transform_window(
    rank='rank(Rate_Family_Income)',
    sort=[alt.SortField('Rate_Family_Income', order='descending')]
).mark_bar().encode(
    y=alt.Y('Family_Income:N', sort='-x', title="Family Income Range"),
    x=alt.X('mean(Diabetes):Q', title="Diabetes Rate(%)", axis=alt.Axis(tickCount=5)),
    color=alt.condition(
        alt.datum.Family_Income == income_type,  
        alt.value('orange'),     # which sets the bar orange.
        alt.value('#F9E79F')),
    tooltip=[alt.Tooltip("Family_Income", title="Family Income Range"),
             alt.Tooltip("mean(Diabetes):Q",title="Diabetes Rate(%)")]
).properties(
    title=f"{year} Family Income vs Prevalence of Diabetes"
)
plot3_income
st.write("Your Family Income Group is highlighted in orange.")



plot3_age = alt.Chart(subset).transform_joinaggregate(
    Rate_Age='mean(Diabetes)',
    groupby=['Age_Group']
).transform_window(
    rank='rank(Rate_Age)',
    sort=[alt.SortField('Rate_Age', order='descending')]
).mark_bar().encode(
    y=alt.Y('Age_Group:N', sort='-x', title="Age Group"),
    x=alt.X('mean(Diabetes):Q', title="Diabetes Rate(%)", axis=alt.Axis(tickCount=5)),
    color=alt.condition(
        alt.datum.Age_Group == age_type,  
        alt.value('orange'),     # which sets the bar orange.
        alt.value('#52BE80')),
    tooltip=[alt.Tooltip("Age_Group", title="Age Group"),
             alt.Tooltip("mean(Diabetes):Q",title="Diabetes Rate(%)")]
).properties(
    title=f"{year} Age Group vs Prevalence Of Diabetes"
)

plot3_age
st.write("Your Age Group is highlighted in orange.")
#########################################PLOT3#########################################



#########################################PLOT2#########################################
# Create DataFrame
st.header("Diabetes rate in different BMI group")

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
plot2 = alt.Chart(plot2df).mark_bar().encode(alt.X('Gender',axis=None),alt.Y('Rate', title='Rate(%)'),color=alt.Color('Gender'),column=alt.Column('BMI_Group', sort=[ 'Underweight', 'Normal', 'Overweight','Obese'], header=alt.Header(titleFontSize=10),title='BMI Group'),tooltip= [alt.Tooltip(field = "Rate",title = "Rate(%)")]).properties(
    title=str(year)+' Diabetes Rate vs BMI').properties(
    width=alt.Step(90) 
)

user_height_m = user_height/100
user_BMI_num=user_weight/(user_height_m*user_height_m)
if user_BMI_num <18.5:
    user_BMI ="Underweight"
if user_BMI_num >= 18.5 and user_BMI_num < 25:
    user_BMI ="Normal"
if user_BMI_num >= 25 and user_BMI_num < 30:
    user_BMI ="Overweight"
if user_BMI_num >= 30:
    user_BMI ="Obese"


line2 =alt.Chart(plot2df).mark_line(shape="stroke",color="blue").encode(
           alt.X('BMI_Group', sort=[ 'Underweight', 'Normal', 'Overweight','Obese'],title='BMI Group'),
           alt.Y('Average',title='Average Rate(%)'),tooltip=[ alt.Tooltip(field = "Average", title = "Average Rate(%)")],
    shape=alt.condition(alt.datum.BMI_Group == user_BMI,alt.value('circle'),    
        alt.value('circle')),
color=alt.condition(alt.datum.BMI_Group == user_BMI,alt.value('red'),     
        alt.value('steelblue'))).properties(
     title=str(year)+' Overall Diabetes Rate of Different BMI Group',
)

plt2=alt.vconcat(plot2, line2).configure_axisX(labelAngle=360).configure_point(
    size=100
)
plt2
st.write("Your BMI group's average diabetes rate is highlighted in RED")
#########################################PLOT2#########################################



#########################################PLOT4#########################################
# Plot4
# Change data from wide to long
st.header("Prevalence of diabetes in population with underlying health conditions")
disease_long = pd.melt(df, id_vars=["Year","Diabetes","Gender"], value_vars=["High_Blood_Pressure","Kidney_Issue","Coronory_Heart_Disease","Thyroid_Issue","Liver_Issue"],var_name="Disease_type")
disease_long = disease_long.dropna(subset=["Disease_type","value"])
disease_long['Disease_type'] = disease_long['Disease_type'].replace(to_replace = ["High_Blood_Pressure","Kidney_Issue","Coronory_Heart_Disease","Thyroid_Issue","Liver_Issue"], value=['High Blood Pressure', 'Kidney Issue', 'Coronory Heart Disease', 'Thyroid Issue', 'Liver Issue'])
disease_long = disease_long[disease_long["value"] != "no"]

# disease multiselect

subset = disease_long[disease_long["Disease_type"].isin(disease)]


disease_selection = alt.selection_single(
    fields=['Disease_type'], 
    bind='legend'
)

# multiple barchart
plot4 = alt.Chart(subset).transform_joinaggregate(
    Rate='mean(Diabetes)',
    groupby=['Disease_type']
).mark_bar(size=18).encode(
    # Tried to move it bottom, but did not work: header=alt.Header(titleOrient='bottom', labelOrient='bottom')
    column=alt.Column('Year:N', title="Year Range", header=alt.Header(labelAngle=-45)),
    x=alt.X("Disease_type:N", axis=alt.Axis(labels=False), title=None),
    y=alt.Y("mean(Diabetes):Q", title="Diabetes Rate(%)"),
    color=alt.Color("Disease_type:N", title="Disease Type"),
    tooltip=[alt.Tooltip("Disease_type", title="Disease Type"),
             alt.Tooltip("mean(Diabetes):Q",title="Diabetes Rate(%)")]
).add_selection(
    disease_selection
).properties(
    title="Diabetes Rate Under Certain Disease"
).properties(
    width=alt.Step(30) 
)

# plot4


year_range = ['2009-2010', '2011-2012', '2013-2014', '2015-2016', '2017-2018']
year_dropdown = alt.binding_select(options=year_range)
year_select = alt.selection_single(fields=['Year'], bind=year_dropdown, name="Select", init={'Year':year_range[0]})

subset['Diabetes_cat'] = subset['Diabetes'].map({100:'Diabetic Population', 0:'Non-diabetic Population'})

pie_chart = alt.Chart(subset
).transform_joinaggregate(
    # num_count_year='count(Disease_type)/100',
    groupby=['Year']
).mark_arc().encode(
     theta = alt.Theta(shorthand='count(Disease_type):Q'),
     color = alt.Color(field='Diabetes_cat', type='nominal', title="Health Condition"),
     tooltip=[alt.Tooltip("count(Disease_type):Q", title="Number of people"),
              alt.Tooltip("Diabetes_cat",title="Diabetes Status")]
).transform_filter(
    disease_selection
).properties(
    title="Diabetic and Non-diabetic Population Breakdown in Population with Selected Medical Conditions From 2009 To 2018",width=200,height=200
    #f"{year} Age Group vs prevalence of diabetes"
)

# chart = alt.hconcat()
# for type in disease:
#     chart = base.transform_filter(datum.Disease_type== type)


final_plot = alt.vconcat(plot4, pie_chart).resolve_scale(
    color='independent'
)

final_plot
st.write("Your selected medical conditions are shown in the bar plot.")
#########################################PLOT4#########################################
