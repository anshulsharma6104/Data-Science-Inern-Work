#!/usr/bin/env python
# coding: utf-8

# # TASK 1: Data Exploration and Preprocessing 

# # DATA EXPLORATION

# ### STEP 1--- IMPORTING THE DATASET

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data_set=pd.read_csv(r"C:\Users\anshu\Dataset .csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ### STEP 2 --- DISPLAYING THE FIRST FEW ROWS

# In[3]:


data_set.head()


# ### STEP 3 --- CHECKING THE DATA TYPE OF EACH COLUMN

# In[4]:


data_set.dtypes


# ###  STEP 4 --- GETTING SUMMARY STATISTICS FOR NUMERICAL COLUMNS

# In[5]:


data_set.describe()


# ### STEP 5 --- GETTING SUMMARY STATISTICS FOR CATEGORICAL COLUMNS

# In[6]:


data_set.describe(include=['O'])


# # IDENTIFYING THE NUMBER OF ROWS AND COLUMNS

# In[7]:


num_rows, num_columns = data_set.shape

print(f"The dataset has {num_rows} rows and {num_columns} columns.")


# # CHECKING FOR MISSING VALUES

# In[8]:


missing_values = data_set.isnull().sum()
print(missing_values)


# ### VISUALIZING THE MISSING VALUES

# In[9]:


plt.figure(figsize=(6, 4))
missing_values.plot(kind='bar')
plt.title('Missing Values per Column')
plt.ylabel('Number of Missing Values')
plt.show()


# ### DISPLAYING LOCATIONS OF THE MISSING VALUES

# In[7]:


pd.set_option('display.max_columns', 5)
rows_with_missing = data_set[data_set.isnull().any(axis=1)]
print(rows_with_missing)


# # HANDLING THE MISSING VALUES

# ### IMPUTATION WITH MODE OF MISSING VALUES

# In[10]:


mode_value = data_set['Cuisines'].mode()[0]
data_set['Cuisines'].fillna(mode_value, inplace=True)


# ### VERIFYING THAT THERE ARE NO MISSING VALUES LEFT

# In[20]:


# Verify that there are no missing values left
missing_values = data_set.isnull().sum()
print(missing_values)
print(f"Missing values in 'Cuisines' after imputation: {data_set['Cuisines'].isnull().sum()}")


# # DATA TYPE CONVERSION

# ### CONVERTING CATEGORICAL DATA INTO 'CATEGORY' DATA TYPE

# In[22]:


categorical_columns = ['Country Code', 'Currency', 'Rating color', 'Rating text']
for column in categorical_columns:
    data_set[column] = data_set[column].astype('category')


# In[26]:


data_set.dtypes


# ### CONVERTING BOOLEAN DATA INTO 'BOOLEAN' DATA TYPE

# In[28]:


boolean_columns = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
for column in boolean_columns:
    data_set[column] = data_set[column].map({'Yes': True, 'No': False})


# In[29]:


data_set.dtypes


# # ANALYZING THE DISTRIBUTION OF THE TARGET VARIABLE 

# ### TARGET VARIABLE IS AGGREGATE RATING

# ### DESCRIPTIVE STATISTICS

# In[36]:


aggregate_rating_stats = data_set['Aggregate rating'].describe()
print(aggregate_rating_stats)


# ### FREQUENCY DISTRIBUTION

# In[42]:


aggregate_rating_value_counts = data_set['Aggregate rating'].value_counts().sort_index()
print(aggregate_rating_value_counts)


# ### PLOTTING THE DISTRIBUTION

# In[39]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(data_set['Aggregate rating'], kde=True, bins=20, color='blue')
plt.title('Distribution of Aggregate Rating')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.show()


# # IDENTIFYING CLASS IMBALANCES IN TARGET VARIABLE

# ### CALCULATING THE PERCENTAGE OF EACH RATING

# In[40]:


aggregate_rating_percentages = (aggregate_rating_value_counts / len(data_set)) * 100
print(aggregate_rating_percentages)


# ### PLOTTING THE FREQUENCY DISTRIBUTION 

# In[43]:


plt.figure(figsize=(10, 6))
sns.barplot(x=aggregate_rating_value_counts.index, y=aggregate_rating_value_counts.values, palette="viridis")
plt.title('Frequency Distribution of Aggregate Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.show()


# ### THE ABOVE DISTRIBUTION INDICATES A CLASS IMBALACE AT 0.0

# # TASK 2 : DESCRIPTIVE ANALYSIS

# ### NUMERICAL COLUMNS IN THE DATASET ARE:

# In[44]:


numerical_columns = data_set.select_dtypes(include=[np.number]).columns.tolist()
print(numerical_columns)


# ### CALCULATING BASIC STATISTICAL NUMERICAL MEASURES FOR EACH COLUMN

# In[77]:


col_data = data_set['column_name']

mean = col_data.mean()
median = col_data.median()
mode = col_data.mode()[0] if not col_data.mode().empty else np.nan
std_dev = col_data.std()
range_value = col_data.max() - col_data.min()
percentiles = col_data.quantile([0.25, 0.5, 0.75]).to_dict()


# ### STATISTICAL MEASURES FOR LONGITUDE

# In[64]:


print(f'Mean: {mean}')
print(f'Median: {median}')
print(f'Mode: {mode}')
print(f'Standard Deviation: {std_dev}')
print(f'Range: {range_value}')
print(f'Percentiles: {percentiles}')


# ### STATISTICAL MEASURES FOR LATITUDE

# In[68]:


print(f'Mean: {mean}')
print(f'Median: {median}')
print(f'Mode: {mode}')
print(f'Standard Deviation: {std_dev}')
print(f'Range: {range_value}')
print(f'Percentiles: {percentiles}')


# ### STATISTICAL MEASURES FOR AVERAGE COST FOR TWO

# In[70]:


print(f'Mean: {mean}')
print(f'Median: {median}')
print(f'Mode: {mode}')
print(f'Standard Deviation: {std_dev}')
print(f'Range: {range_value}')
print(f'Percentiles: {percentiles}')


# ### STATISTICAL MEASURES FOR PRICE RANGE

# In[72]:


print(f'Mean: {mean}')
print(f'Median: {median}')
print(f'Mode: {mode}')
print(f'Standard Deviation: {std_dev}')
print(f'Range: {range_value}')
print(f'Percentiles: {percentiles}')


# ### STATISTICAL MEASURES FOR AGGREGATE RATING

# In[75]:


print(f'Mean: {mean}')
print(f'Median: {median}')
print(f'Mode: {mode}')
print(f'Standard Deviation: {std_dev}')
print(f'Range: {range_value}')
print(f'Percentiles: {percentiles}')


# ### STATISTICAL MEASURES FOR VOTES

# In[78]:


print(f'Mean: {mean}')
print(f'Median: {median}')
print(f'Mode: {mode}')
print(f'Standard Deviation: {std_dev}')
print(f'Range: {range_value}')
print(f'Percentiles: {percentiles}')


# # EXPLORING THE DISTRIBUTION OF CATEGORICAL VARIABLES

# ### FOR COUNTRY CODE

# ### CHECKING UNIQUE VALUES AND THEIR COUNT FOR 'COUNTRY CODE'

# In[93]:


country_counts = data_set['Country Code'].value_counts()
print(country_counts)


# ### DISTRIBUTION OF COUNTRY CODE USING BAR PLOT

# In[178]:


plt.figure(figsize=(12, 6))

sns.barplot(x=country_counts.index, y=country_counts.values, palette='viridis')
plt.yscale('log')
plt.title('Distribution of Country Code (Log Scale)')
plt.xlabel('Country Code')
plt.ylabel('Frequency (Log Scale)')
plt.xticks(rotation=90)
    
plt.show()


# ### DISTRIBUTION OF COUNTRY CODE USING PIE CHART

# In[120]:


threshold = 100

frequent_countries = country_counts[country_counts >= threshold]
infrequent_countries = country_counts[country_counts < threshold]

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

axes[0].pie(frequent_countries, labels=frequent_countries.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(frequent_countries)))
axes[0].set_title('Distribution of Frequent Country Codes')
axes[0].axis('equal')  

axes[1].pie(infrequent_countries, labels=infrequent_countries.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(infrequent_countries)))
axes[1].set_title('Distribution of Infrequent Country Codes')
axes[1].axis('equal')

plt.tight_layout()
plt.show()


# ### FOR CITY

# ### CHECKING UNIQUE VALUES AND THEIR COUNT FOR 'CITY'

# In[133]:


city_counts = data_set['City'].value_counts()
print(city_counts)


# ### DISTRIBUTION OF CITY USING BAR PLOT

# In[162]:


data_set['City_Grouped'] = data_set['City']

data_set.loc[data_set['City'].isin(city_counts[city_counts < threshold].index), 'City_Grouped'] = 'Others'

grouped_city_counts = data_set['City_Grouped'].value_counts()

grouped_city_counts = grouped_city_counts.sort_values(ascending=False)
if 'Others' in grouped_city_counts.index:
    others_value = grouped_city_counts.pop('Others')
    grouped_city_counts['Others'] = others_value

plt.figure(figsize=(12, 6))
sns.barplot(x=grouped_city_counts.index, y=grouped_city_counts.values, palette='viridis')
plt.title('Distribution of City (Grouped)')
plt.xlabel('City')
plt.ylabel('Frequency')
plt.xticks(rotation=0)

for index, value in enumerate(grouped_city_counts.values):
    plt.text(index, value + 50, str(value), ha='center', va='bottom')

plt.show()


# ### DISTRIBUTION OF CITY USING PIE CHART

# In[148]:


threshold = 100

data_set['City_Grouped'] = data_set['City']

data_set.loc[data_set['City'].isin(city_counts[city_counts < threshold].index), 'City_Grouped'] = 'Others'

grouped_city_counts = data_set['City_Grouped'].value_counts()

grouped_city_counts = grouped_city_counts.sort_values(ascending=False)
if 'Others' in grouped_city_counts.index:
    others_value = grouped_city_counts.pop('Others')
    grouped_city_counts['Others'] = others_value

plt.figure(figsize=(7,7))
plt.pie(grouped_city_counts, labels=grouped_city_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(grouped_city_counts)))
plt.title('Distribution of City (Grouped)')
plt.axis('equal')
plt.show()


# ### FOR CUISINES

# In[152]:


cuisine_counts = data_set['Cuisines'].value_counts()
print(cuisine_counts)


# ### DISTRIBUTION OF CUISINES USING BAR PLOT

# In[161]:


threshold = 100

data_set['Cuisines_Grouped'] = data_set['Cuisines']

data_set.loc[data_set['Cuisines'].isin(cuisine_counts[cuisine_counts < threshold].index), 'Cuisines_Grouped'] = 'Others'

grouped_cuisine_counts = data_set['Cuisines_Grouped'].value_counts()

grouped_cuisine_counts = grouped_cuisine_counts.sort_values(ascending=False)
if 'Others' in grouped_cuisine_counts.index:
    others_value = grouped_cuisine_counts.pop('Others')
    grouped_cuisine_counts['Others'] = others_value

plt.figure(figsize=(14, 7))
sns.barplot(x=grouped_cuisine_counts.index, y=grouped_cuisine_counts.values, palette='viridis')
plt.title('Distribution of Cuisines (Grouped)')
plt.xlabel('Cuisines')
plt.ylabel('Frequency')
plt.xticks(rotation=90)

for index, value in enumerate(grouped_cuisine_counts.values):
    plt.text(index, value + 50, str(value), ha='center', va='bottom')

plt.show()


# ### DISTRIBUTION OF CUISINES USING PIE CHART

# In[176]:


threshold = 200

data_set['Cuisines_Grouped'] = data_set['Cuisines']

data_set.loc[data_set['Cuisines'].isin(cuisine_counts[cuisine_counts < threshold].index), 'Cuisines_Grouped'] = 'Others'

grouped_cuisine_counts = data_set['Cuisines_Grouped'].value_counts()

grouped_cuisine_counts = grouped_cuisine_counts.sort_values(ascending=False)
if 'Others' in grouped_cuisine_counts.index:
    others_value = grouped_cuisine_counts.pop('Others')
    grouped_cuisine_counts['Others'] = others_value



plt.figure(figsize=(12, 12))
colors = sns.color_palette('viridis', len(grouped_cuisine_counts))
plt.pie(grouped_cuisine_counts, labels=grouped_cuisine_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=140, pctdistance=0.85, labeldistance=1.1)
plt.title('Distribution of Cuisines (Grouped)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.legend(title='Cuisines', bbox_to_anchor=(1.05, 1), loc='best')
plt.show()


# # IDENTIFYING TOP CUISINES

# ### DISPLAYING THE TOP 10 CUISINES

# In[179]:


top_cuisines = cuisine_counts.head(10)
print("Top Cuisines with the highest number of restaurants:")
print(top_cuisines)


# ### PLOTTING THE TOP 10 CUISINES

# In[193]:


plt.figure(figsize=(12, 6))
sns.barplot(x=top_cuisines.index, y=top_cuisines.values, palette='viridis')
plt.title('Top 10 Cuisines with the Highest Number of Restaurants')
plt.xlabel('Cuisines')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.show()


# # IDENTIFYING TOP CITIES

# ### DISPLAYING THE TOP CITIES WITH HIGHEST NUMBER OF RESTAURANTS

# In[200]:


top_cities = city_counts.head(4)
print("Top Cities with the highest number of restaurants:")
print(top_cities)


# ### PLOTTING THE TOP CITIES WITH HIGHEST NUMBER OF RESTAURANTS

# In[201]:


plt.figure(figsize=(12, 6))

sns.barplot(x=top_cities.index, y=top_cities.values, palette='viridis')
plt.title('Top 10 Cities with the Highest Number of Restaurants')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)

plt.show()


# # TASK 3 : GEOSPATIAL ANALYSIS

# In[207]:


pip install folium


# In[206]:


pip install ruamel-yaml


# In[8]:


from folium.plugins import MarkerCluster


# In[5]:


import folium
a_list=data_set[['Restaurant Name','Latitude','Longitude']].values.tolist()
a_list


# # Visualizing the locations of restaurants on a map using latitude and longitude information.

# In[9]:


restaurant_map = folium.Map(location=[14.565443, 121.027535])

marker_cluster = MarkerCluster().add_to(restaurant_map)

for i in a_list:
    folium.Marker(location=[i[1], i[2]], popup=i[0], icon=folium.Icon(color='green')).add_to(marker_cluster)

restaurant_map.add_child(marker_cluster)

restaurant_map.save('restaurants_map_clustered.html')

restaurant_map


# # Analyzing the distribution of restaurants across different cities or countries.

# ### Aggregating the Number of Restaurants by City and Country.

# In[11]:


city_counts = data_set['City'].value_counts()

country_counts = data_set['Country Code'].value_counts()

print("Top 10 Cities with the highest number of restaurants:")
print(city_counts.head(10))

print("\nTop 10 Countries with the highest number of restaurants:")
print(country_counts.head(10))


# ### ANALYZING ACROSS CITIES

# In[12]:


city_threshold = 100

data_set['City_Grouped'] = data_set['City'].apply(lambda x: x if city_counts[x] >= city_threshold else 'Others')

city_grouped_counts = data_set['City_Grouped'].value_counts()

city_grouped_counts = city_grouped_counts.sort_values(ascending=False)
if 'Others' in city_grouped_counts.index:
    others_value = city_grouped_counts.pop('Others')
    city_grouped_counts['Others'] = others_value

plt.figure(figsize=(12, 6))
sns.barplot(x=city_grouped_counts.index, y=city_grouped_counts.values, palette='viridis')
plt.title('Distribution of Restaurants Across Cities')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)

max_count = city_grouped_counts.values.max()
plt.yticks(range(0, max_count + 500, 500))

for index, value in enumerate(city_grouped_counts.values):
    plt.text(index, value + 50, str(value), ha='center', va='bottom')

plt.show()


# ### ANALYZING ACROSS COUNTRIES

# In[21]:


country_threshold = 50

data_set['Country_Grouped'] = data_set['Country Code'].apply(lambda x: x if country_counts[x] >= country_threshold else 'Others')

country_grouped_counts = data_set['Country_Grouped'].value_counts()

country_grouped_counts = country_grouped_counts.sort_values(ascending=False)
if 'Others' in country_grouped_counts.index:
    others_value = country_grouped_counts.pop('Others')
    country_grouped_counts['Others'] = others_value

plt.figure(figsize=(12, 6))
sns.barplot(x=country_grouped_counts.index, y=country_grouped_counts.values, palette='viridis')
plt.title('Distribution of Restaurants Across Countries')
plt.xlabel('Country Code')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)

max_count = country_grouped_counts.values.max()
plt.yticks(range(0, max_count + 50, 500))

for index, value in enumerate(country_grouped_counts.values):
    plt.text(index, value + 5, str(value), ha='center', va='bottom')

plt.show()


# #  CORRELATION BETWEEN THE RESTAURANT'S LOCATION AND ITS RATING.

# ### CHECKING FOR MISSING VALUES IN 'LATITUDE', 'LONGITUDE', AND 'AGGREGATE RATING'

# In[29]:


missing_values = data_set[['Latitude', 'Longitude', 'Aggregate rating']].isnull().sum()
print("Missing values in relevant columns:")
print(missing_values)

data_set_clean = data_set.dropna(subset=['Latitude', 'Longitude', 'Aggregate rating'])


# ### COMPUTING THE CORRELATION MATRIX

# In[30]:


correlation_matrix = data_set_clean[['Latitude', 'Longitude', 'Aggregate rating']].corr()
print("Correlation matrix:")
print(correlation_matrix)


# ### SCATTER PLOT FOR LATITUDE v/s AGGREGATE RATING

# In[35]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Latitude', y='Aggregate rating', data=data_set_clean)
plt.title('Latitude vs Aggregate Rating')
plt.xlabel('Latitude')
plt.ylabel('Aggregate Rating')
plt.show()


# ### SCATTER PLOT FOR LONGITUDE v/s AGGREGATE RATING¶

# In[33]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Longitude', y='Aggregate rating', data=data_set_clean)
plt.title('Longitude vs Aggregate Rating')
plt.xlabel('Longitude')
plt.ylabel('Aggregate Rating')
plt.show()


# ### HEATMAP FOR THE CORRELATION MATRIX

# In[36]:


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# # ANALYSIS OF THE CORRELATION

# ### THE CORRELATION COEFFICIENTS BETWEEN 'LATITUDE' AND 'AGGREGATE RATING', AND 'LONGITUDE' AND 'AGGREGATE RATING' ARE VERY CLOSE TO 0, IMPLYING THAT THERE IS NO SIGNIFICANT LINEAR RELATIONSHIP. THIS MEANS THAT A RESTAURANT'S LOCATION DOES NOT HAVE A MEANINGFUL IMPACT ON ITS RATING BASED ON THE GIVEN DATASET. 
# 

# ### SO THERE IS NO CORRELATION BETWEEN THE RESTAURANT'S LOCATION(LATITUDE & LONGITUDE) AND ITS RATING.
