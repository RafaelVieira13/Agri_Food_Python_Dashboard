import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import pycountry

# Dashboard Title
st.set_page_config(page_title='Agri-Food CO2 Emission', page_icon=':dash:', layout='wide')
st.title(' :dash: Agri-Food CO2 Emission')
st.markdown('<style>div.block-container{padding-top:1rem}<style>',unsafe_allow_html=True)

# Let´s Read the Dataset
df = pd.read_csv('Agrofood_co2_emission.csv')

# Get a Continent Column
def get_continent(country):
    continent = {
        'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Côte d\'Ivoire', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'São Tomé and Príncipe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'],
        'Asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Russia', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen'],
        'Europe': ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican City'],
        'North America': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'],
        'Oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'],
        'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']
    }

    for continent, countries in continent.items():
        if country in countries:
            return continent
    return None

df["continent"] = df["Area"].apply(get_continent)

    # Let´s Only Use Data That We Known The Continents 
df.dropna(subset='continent', inplace=True)

# Let´s apply MinMaxScaler to the data in order to get only positive values
minmax = MinMaxScaler(feature_range=(0,1))
data = df['total_emission'].values.reshape(-1, 1)
df['total_emission_normalized'] = minmax.fit_transform(data)

# Let´s Create a Slide Bar To Select the Year We Want To See
st.subheader('Year Slider:')

min_year = df['Year'].min()
max_year = df['Year'].max()

year_range = st.slider('Select a Year Range Between {} and {}'.format(min_year, max_year),
              min_value=min_year, max_value=max_year, value=(min_year, max_year))

df = df[ (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

# Create a side bar to filter the data based on the Area and Continent
st.sidebar.header('Choose Your Filter: ')

    # Area
area = st.sidebar.multiselect('Pick the Countries', df['Area'].unique())
if not area:
    df2 = df.copy()
else:
    df2 = df[df['Area'].isin(area)]

    # Continent
continent = st.sidebar.multiselect('Pick The Continets', df['continent'].unique())
if not continent:
    df3 = df2.copy()
else:
    df3 = df2[df2['continent'].isin(continent)]

# Filter The Data Based On the Side Bar
if not area and not continent:
    filtered_df = df
elif not area:
    filtered_df = df[df['continent'].isin(continent)]
elif not continent:
    filtered_df = df[df['Area'].isin(area)]
else: 
    filtered_df = df3[(df3['Area'].isin(area)) & (df3['continent'].isin(continent))]

# Normalize Data (Get Only Positive and Negative Values) - Usefull To Create The Map
minmax = MinMaxScaler(feature_range=(0,1))
data = filtered_df['total_emission'].values.reshape(-1, 1)
filtered_df['total_emission_normalized'] = minmax.fit_transform(data)

# Get the Countries ISO - Usefull to Create the Map
def get_iso_alpha(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        iso_alpha = country.alpha_3
        return iso_alpha
    except:
        return None
    
filtered_df['iso_alpha'] = filtered_df['Area'].apply(get_iso_alpha)

# Round Data
filtered_df['total_emission'] = filtered_df['total_emission'].round(2)
filtered_df['total_emission_normalized'] = filtered_df['total_emission_normalized'].round(2)

# Map To Display the CO2 Emission
st.subheader('Agrifood CO2 emission by Country')
fig = px.scatter_geo(filtered_df, 
                        locations="iso_alpha", 
                        color='continent',
                        hover_name="Area", 
                        hover_data=["Area","total_emission"],
                        size="total_emission_normalized"
                        )
st.plotly_chart(fig, use_container_width=True)

# Line plot to get the total_emission by year
average_emssion_by_year = filtered_df.groupby('Year').agg(total_emission_average=('total_emission','mean')).reset_index()
average_emssion_by_year['total_emission_average'] = average_emssion_by_year['total_emission_average'].round(2)

st.subheader('Average Total CO2 By Year')
fig = px.line(average_emssion_by_year, x = 'Year', y = 'total_emission_average')
fig.update_layout( xaxis_title = 'Year', yaxis_title = 'CO2 Emission')
st.plotly_chart(fig, use_container_width=True)

# Bar-Plot to Display the Average CO2 Emission Reasons
    # Select only the numeric columns
numeric_columns = filtered_df.iloc[1:-1].select_dtypes(include=[int, float])
    # Perform groupby and calculate the mean for each numeric column for each 'Area'
reason_co2_mean = filtered_df.agg({col: 'mean' for col in numeric_columns.columns}).reset_index()
    # Rename Columns
reason_co2_mean  = reason_co2_mean.rename(columns = {'index':'Reason',0:'Average CO2 Emissions'})
    # Remove Data That Doesn´t Matter
reason_co2_mean = reason_co2_mean.drop([0,8,24,25,26,27,28,29,30])
    # Let´s round the 'Average CO2 Emissions' to 0 decimals places
reason_co2_mean['Average CO2 Emissions'] = reason_co2_mean['Average CO2 Emissions'].round(0)
    # Create a Barplot
st.subheader('Average CO2 Emission by Reason')
fig = px.bar(reason_co2_mean, x = 'Reason', y = 'Average CO2 Emissions')
fig.update_layout(xaxis_title = 'Reason', 
                  yaxis_title = 'Average CO2 Emissions')
    # Let´s add the symbol kt
fig.update_traces(hovertemplate='<b>Reason: %{x}</b><br><b>Average CO2 Emission: %{y:.2f} kt</b>',
                  marker={'color': 'rgb(158, 202, 225)'})
st.plotly_chart(fig, use_container_width=True)

# Relasionship Between Population and CO2 Emission and Temperature and CO2 Emission -- Scatter Plot
    # Let´s get total population data
filtered_df['Total Population'] = filtered_df['Total Population - Male'] + filtered_df['Total Population - Female']

col1, col2 = st.columns(2)

with col1:
    st.subheader('Total Population vs CO2 Emission')
    fig = px.scatter(filtered_df, x = 'Total Population', y = 'total_emission')
    fig.update_layout(xaxis_title = 'Total Population', yaxis_title = 'CO2 Emission',template="plotly_dark")
    st.plotly_chart(fig,use_container_width=True)

with col2:
    st.subheader('Temperature °C vs CO2 Emission')
    fig = px.scatter(filtered_df, x = 'Average Temperature °C', y = 'total_emission')
    fig.update_layout(xaxis_title = 'Average Temperature °C', yaxis_title = 'CO2 Emission',template="plotly_dark")
    st.plotly_chart(fig,use_container_width=True)
    
# Let´s Create Two Pie Charts In Order to Analyse the Average Population By Year and By Gender
    # Let´s manipulate the data
filtered_df['Total Population'] = filtered_df['Total Population - Male'] + filtered_df['Total Population - Female']
    # Let´s Get Demographic Data
demographic = filtered_df[['Rural population','Urban population','Total Population']]
demographic = pd.DataFrame(demographic.mean()).reset_index()
demographic = demographic.rename(columns={'index':'Demographic',0:'Average Population'})
demographic['Population %'] = demographic['Average Population'] / filtered_df['Total Population'].mean() *100
demographic['Population %'] = demographic['Population %'].round(1)
demographic = demographic.drop([2])
    # Let´s Get Gender Data
gender = filtered_df[['Total Population - Male','Total Population - Female','Total Population']]
gender = pd.DataFrame(gender.mean()).reset_index()
gender = gender.rename(columns={'index':'Gender',0:'Average Population'})
gender['Population %'] = gender['Average Population'] / filtered_df['Total Population'].mean() *100
gender['Population %'] = gender['Population %'].round(1)
gender = gender.drop([2])
gender['Gender'] = gender['Gender'].str.replace('Total Population - Male','Male')
gender['Gender'] = gender['Gender'].str.replace('Total Population - Female','Female')

    # Let´s Buld 4 Pie Charts
col1, col2, = st.columns(2)

with col1:
    st.subheader('Average Population By Demographic')
    fig = px.pie(demographic, values= 'Average Population', names = 'Demographic', template = 'plotly_dark')
    fig.update_traces(text = demographic['Demographic'], textposition = 'inside')
    st.plotly_chart(fig,use_container_width=True)

with col2:
    st.subheader('Average Population By Gender')
    fig = px.pie(gender, values= 'Average Population', names = 'Gender', template = 'gridon')
    fig.update_traces(text = gender['Gender'], textposition = 'inside')
    st.plotly_chart(fig,use_container_width=True)











