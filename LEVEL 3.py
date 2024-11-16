#!/usr/bin/env python
# coding: utf-8

# In[1]:


crew install pandoc


# # TASK 1 : PREDICTIVE MODELING

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
file_path = 'Data_set.csv'
dataset = pd.read_csv(file_path)


# In[4]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
dataset.head()


# ###  BUILDING A REGRESSION MODEL TO PREDICT THE AGGREGATE RATING OF A RESTAURANT BASED ON AVAILABLE FEATURES.

# ### HERE WE HAVE ALSO SPLITTED THE DATASET INTO TRAINING AND TESTING SETS AND NOW WE WILL EVALUATE THE MODEL'S PERFORMANCE USING APPROPRIATE METRICS.

# ### LINEAR REGRESSION

# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = dataset.drop(columns=['Aggregate rating', 'Rating color', 'Rating text'])
y = dataset['Aggregate rating']

categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder().fit(X[col]) for col in categorical_cols}

for col, encoder in label_encoders.items():
    X[col] = encoder.transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)
y_pred_lr = linear_regression.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - Mean Squared Error: {mse_lr}, R² Score: {r2_lr}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, color='blue')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Linear Regression')


# ### DECISION TREE ALGORITHM

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import mean_squared_error, r2_score

X = dataset.drop(columns=['Aggregate rating', 'Rating color', 'Rating text'])
y = dataset['Aggregate rating']

categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder().fit(X[col]) for col in categorical_cols}

for col, encoder in label_encoders.items():
    X[col] = encoder.transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

decision_tree = DecisionTreeRegressor(random_state=42)

decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Decision Tree - Mean Squared Error: {mse_dt}, R² Score: {r2_dt}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_dt, alpha=0.6, color='green')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Decision Tree')


# ### RANDOM FOREST ALGORITHM

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import mean_squared_error, r2_score

X = dataset.drop(columns=['Aggregate rating', 'Rating color', 'Rating text'])
y = dataset['Aggregate rating']

categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder().fit(X[col]) for col in categorical_cols}

for col, encoder in label_encoders.items():
    X[col] = encoder.transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - Mean Squared Error: {mse_rf}, R² Score: {r2_rf}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='red')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Random Forest')


# ### PLOTTING ALL THREE MODELS SIDE BY SIDE

# In[14]:


plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.6, color='blue')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Linear Regression')

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_dt, alpha=0.6, color='green')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Decision Tree')

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='red')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Random Forest')

plt.tight_layout()
plt.show()


# ### COMPARING THE PERFORMANCE OF THE THREE ALGORITHMS

# In[15]:


models = ['Linear Regression', 'Decision Tree', 'Random Forest']
mse_scores = [mse_lr, mse_dt, mse_rf]
r2_scores = [r2_lr, r2_dt, r2_rf]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('Mean Squared Error', color=color)
ax1.bar(models, mse_scores, color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('R² Score', color=color)
ax2.plot(models, r2_scores, color=color, marker='o', linewidth=2, markersize=6)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Model Performance Comparison')
plt.show()


# # TASK 2 : CUSTOMER PREFERENCE ANALYSIS

# ### ANALYZING THE RELATIONSHIP BETWEEN THE TYPE OF CUISINE AND THE RESTAURANT'S RATING.

# In[27]:


dataset['Cuisines'] = dataset['Cuisines'].str.split(', ')
dataset = dataset.explode('Cuisines')

cuisine_rating = dataset.groupby('Cuisines')['Aggregate rating'].mean().reset_index()

cuisine_rating = cuisine_rating.sort_values(by='Aggregate rating', ascending=False)

top_cuisines = cuisine_rating.head(20)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_cuisines, x='Aggregate rating', y='Cuisines', palette='viridis')

plt.title('Top 20 Cuisines by Average Restaurant Rating')
plt.xlabel('Average Rating')
plt.ylabel('Cuisine Type')
plt.xticks(range(0, 6))

plt.show()


# ### IDENTIFYING THE MOST POPULAR CUISINES AMONG CUSTOMERS BASED ON THE NUMBER OF VOTES

# In[28]:


dataset['Cuisines'] = dataset['Cuisines'].str.split(', ')
dataset = dataset.explode('Cuisines')

dataset['Votes'] = pd.to_numeric(dataset['Votes'], errors='coerce')

cuisine_votes = dataset.groupby('Cuisines')['Votes'].sum().reset_index()

cuisine_votes = cuisine_votes.sort_values(by='Votes', ascending=False)

top_cuisines_votes = cuisine_votes.head(20)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_cuisines_votes, x='Votes', y='Cuisines', palette='viridis')

plt.title('Top 20 Cuisines by Total Number of Votes')
plt.xlabel('Total Number of Votes')
plt.ylabel('Cuisine Type')

plt.show()


# ### DETERMINING IF THERE ARE SPECIFIC CUISINES THAT TEND TO RECEIVE HIGHER RATING

# In[29]:


dataset['Cuisines'] = dataset['Cuisines'].str.split(', ')
dataset = dataset.explode('Cuisines')


cuisine_rating = dataset.groupby('Cuisines')['Aggregate rating'].mean().reset_index()

cuisine_rating = cuisine_rating.sort_values(by='Aggregate rating', ascending=False)

top_cuisines_rating = cuisine_rating.head(20)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_cuisines_rating, x='Aggregate rating', y='Cuisines', palette='viridis')

plt.title('Top 20 Cuisines by Average Restaurant Rating')
plt.xlabel('Average Rating')
plt.ylabel('Cuisine Type')
plt.xticks(range(0, 6))

plt.show()


# ### Yes, based on the analysis, there are specific cuisines that tend to receive higher ratings. Here are the top 20 cuisines by average restaurant rating:

# In[ ]:


Sunda: 4.90 
Börek: 4.70 
Taiwanese: 4.65
Ramen: 4.50
Dim Sum: 4.47
Hawaiian: 4.41
Döner: 4.40 
Bubble Tea: 4.40
Curry: 4.40
Kebab: 4.38
Izgara: 4.35
Filipino: 4.34
Scottish: 4.33
South African: 4.33
Turkish Pizza: 4.33
World Cuisine: 4.30
Fish and Chips: 4.30
Gourmet Fast Food: 4.30
Kiwi: 4.30
Durban: 4.30


# # DATA VISUALIZATION

# ### CREATING VISUALIZATIONS TO REPRESENT THE DISTRIBUTION OF RATINGS USING DIFFERENT CHARTS

# ### USING HISTOGRAM

# In[11]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Aggregate rating'], bins=10, kde=True, color='blue')
plt.title('Distribution of Aggregate Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.show()


# ### USING BAR PLOT

# In[12]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
rating_counts = dataset['Aggregate rating'].value_counts().sort_index()
rating_counts_excluding_zero = rating_counts[rating_counts.index != 0]
sns.barplot(x=rating_counts_excluding_zero.index, y=rating_counts_excluding_zero.values, palette='viridis')
plt.title('Bar Plot of Aggregate Ratings (Excluding 0 Ratings)')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.show()


# ### USING BOX PLOT

# In[13]:


sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))
top_cuisines = dataset['Cuisines'].value_counts().index[:10]
filtered_dataset = dataset[dataset['Cuisines'].isin(top_cuisines)]
sns.boxplot(x='Aggregate rating', y='Cuisines', data=filtered_dataset, palette='viridis')
plt.title('Box Plot of Aggregate Ratings by Cuisine')
plt.xlabel('Aggregate Rating')
plt.ylabel('Cuisine')
plt.show()


# ### USING VIOLIN PLOT

# In[14]:


sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))
sns.violinplot(x='Aggregate rating', y='Cuisines', data=filtered_dataset, palette='viridis', inner='quartile')
plt.title('Violin Plot of Aggregate Ratings by Cuisine')
plt.xlabel('Aggregate Rating')
plt.ylabel('Cuisine')
plt.show()


# ### COMPARING THE AVERAGE RATINGS OF DIFFERENT CUISINES 

# In[21]:


sns.set(style="whitegrid")

cuisine_ratings = dataset.groupby('Cuisines')['Aggregate rating'].mean().reset_index()

top_cuisine_ratings = cuisine_ratings.sort_values(by='Aggregate rating', ascending=False).head(10)

plt.figure(figsize=(14, 8))
sns.barplot(x='Aggregate rating', y='Cuisines', data=top_cuisine_ratings, palette='viridis')
plt.title('Average Ratings by Cuisine (Top 10)')
plt.xlabel('Average Rating')
plt.ylabel('Cuisine Types')
plt.show()


# ### COMPARING THE AVERAGE RATINGS OF DIFFERENT CITIES

# In[16]:


city_ratings = dataset.groupby('City')['Aggregate rating'].mean().reset_index()
top_city_ratings = city_ratings.sort_values(by='Aggregate rating', ascending=False).head(10)

plt.figure(figsize=(14, 8))
sns.barplot(x='Aggregate rating', y='City', data=top_city_ratings, palette='viridis')
plt.title('Average Ratings by City (Top 10)')
plt.xlabel('Average Rating')
plt.ylabel('City')
plt.show()


# ### VISUALIZING THE RELATIONSHIP BETWEEN VARIOUS FEATURES AND THE TARGET VARIABLES TO GAIN INSIGHTS

# ### SCATTER PLOT : VOTES vs AGGREGATE RATING

# In[23]:


dataset_cleaned = dataset.dropna(subset=['Cuisines', 'City', 'Aggregate rating', 'Votes', 'Average Cost for two'])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Votes', y='Aggregate rating', data=dataset_cleaned, hue='Aggregate rating', palette='viridis')
plt.title('Votes vs Aggregate Rating')
plt.xlabel('Votes')
plt.ylabel('Aggregate Rating')
plt.show()


# ### SCATTER PLOT : AVERAGE COST FOR TWO vs AGGREGATE RATING

# In[35]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average Cost for two', y='Aggregate rating', data=dataset_cleaned, hue='Aggregate rating', palette='viridis')
plt.title('Average Cost for Two vs Aggregate Rating')
plt.xlabel('Average Cost for Two')
plt.ylabel('Aggregate Rating')
plt.show()


# ### BOX PLOT : AGGREGATE RATING BY CITY

# In[26]:


plt.figure(figsize=(14, 8))
top_cities = dataset_cleaned['City'].value_counts().index[:10]
filtered_dataset_city = dataset_cleaned[dataset_cleaned['City'].isin(top_cities)]
sns.boxplot(x='Aggregate rating', y='City', data=filtered_dataset_city, palette='viridis')
plt.title('Aggregate Rating by City (Top 10)')
plt.xlabel('Aggregate Rating')
plt.ylabel('City')
plt.show()


# ### BOX PLOT : AGGREGATE RATING BY CUISINE 

# In[27]:


plt.figure(figsize=(14, 8))
top_cuisines = dataset_cleaned['Cuisines'].value_counts().index[:10]
filtered_dataset_cuisine = dataset_cleaned[dataset_cleaned['Cuisines'].isin(top_cuisines)]
sns.boxplot(x='Aggregate rating', y='Cuisines', data=filtered_dataset_cuisine, palette='viridis')
plt.title('Aggregate Rating by Cuisine (Top 10)')
plt.xlabel('Aggregate Rating')
plt.ylabel('Cuisine')
plt.show()


# ### HEATMAP : CORRELATION MATRIX

# In[28]:


plt.figure(figsize=(10, 8))
correlation_matrix = dataset_cleaned[['Aggregate rating', 'Votes', 'Average Cost for two']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# ### PAIR PLOT

# In[29]:


sns.pairplot(dataset_cleaned[['Aggregate rating', 'Votes', 'Average Cost for two']], hue='Aggregate rating', palette='viridis')
plt.show()


# ### BAR PLOT : AVERAGE RATING BY CITY

# In[30]:


city_ratings = dataset_cleaned.groupby('City')['Aggregate rating'].mean().reset_index()
top_city_ratings = city_ratings.sort_values(by='Aggregate rating', ascending=False).head(10)
plt.figure(figsize=(14, 8))
sns.barplot(x='Aggregate rating', y='City', data=top_city_ratings, palette='viridis', ci=None)
for index, value in enumerate(top_city_ratings['Aggregate rating']):
    plt.text(value, index, f' {value:.2f}', color='black', ha="left", va="center")
plt.title('Average Rating by City (Top 10)')
plt.xlabel('Average Rating')
plt.ylabel('City')
plt.show()

