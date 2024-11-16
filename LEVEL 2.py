#!/usr/bin/env python
# coding: utf-8

# # TASK 1 : TABLE BOOKING AND ONLINE DELIVERY

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset=pd.read_csv('Dataset.csv')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# ### DETERMINING THE PERCENTAGE OF RESTAURANTS THAT OFFER TABLE BOOKING AND ONLINE DELIVERY.

# In[9]:


dataset['Has Table booking'] = dataset['Has Table booking'].str.lower()
dataset['Has Online delivery'] = dataset['Has Online delivery'].str.lower()

total_restaurants = len(dataset)

table_booking_count = dataset[dataset['Has Table booking'] == 'yes'].shape[0]
online_delivery_count = dataset[dataset['Has Online delivery'] == 'yes'].shape[0]

table_booking_percentage = (table_booking_count / total_restaurants) * 100
online_delivery_percentage = (online_delivery_count / total_restaurants) * 100

print(f"Percentage of restaurants that offer table booking: {table_booking_percentage:.2f}%")
print(f"Percentage of restaurants that offer online delivery: {online_delivery_percentage:.2f}%")


# ### VISUALIZING THE PERCENTAGE OF RESTAURANTS THAT OFFER TABLE BOOKING AND ONLINE DELIVERY.

# In[17]:


table_booking_counts = dataset['Has Table booking'].value_counts(normalize=True) * 100
online_delivery_counts = dataset['Has Online delivery'].value_counts(normalize=True) * 100

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.pie(table_booking_counts, labels=table_booking_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral'])
plt.title('Percentage of Restaurants Offering Table Booking')

plt.subplot(1, 2, 2)
plt.pie(online_delivery_counts, labels=online_delivery_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral'])
plt.title('Percentage of Restaurants Offering Online Delivery')

plt.tight_layout()
plt.show()


# ### COMPARING THE AVERAGE RATINGS OF RESTAURANTS WITH TABLE BOOKING AND THOSE WITHOUT.

# In[12]:


dataset['Has Table booking'] = dataset['Has Table booking'].str.lower()

average_rating_with_booking = dataset[dataset['Has Table booking'] == 'yes']['Aggregate rating'].mean()
average_rating_without_booking = dataset[dataset['Has Table booking'] == 'no']['Aggregate rating'].mean()

print(f"Average rating of restaurants with table booking: {average_rating_with_booking:.2f}")
print(f"Average rating of restaurants without table booking: {average_rating_without_booking:.2f}")


# ### VISUALIZING THE AVERAGE RATINGS OF RESTAURANTS WITH TABLE BOOKING AND THOSE WITHOUT.

# In[19]:


avg_ratings_table_booking = dataset.groupby('Has Table booking')['Aggregate rating'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Has Table booking', y='Aggregate rating', data=avg_ratings_table_booking, palette='viridis')

plt.xlabel('Has Table Booking')
plt.ylabel('Average Aggregate Rating')
plt.title('Average Ratings of Restaurants With and Without Table Booking')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.ylim(0, 5)
plt.show()


# ### ANALYZE THE AVAILABILITY OF ONLINE DELIVERY AMONG RESTAURANTS WITH DIFFERENT PRICE RANGES.

# In[13]:


dataset['Has Online delivery'] = dataset['Has Online delivery'].str.lower()

online_delivery_by_price = dataset.groupby('Price range')['Has Online delivery'].value_counts(normalize=True).unstack().fillna(0)

print(online_delivery_by_price)

import matplotlib.pyplot as plt

online_delivery_by_price.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Availability of Online Delivery by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Proportion of Restaurants')
plt.legend(title='Has Online delivery')
plt.xticks(rotation=0)
plt.show()


# # TASK 2 : PRICE RANGE ANALYSIS

# ### DETERMINE THE MOST COMMON PRICE RANGE AMONG ALL THE RESTAURANTS.

# In[21]:


most_common_price_range = dataset['Price range'].value_counts().idxmax()

print(f"The most common price range is: {most_common_price_range}")

price_range_counts = dataset['Price range'].value_counts()
print(price_range_counts)


# ### VISUALIZING THE MOST COMMON PRICE RANGE AMONG ALL THE RESTAURANTS.

# In[16]:


price_range_counts = dataset['Price range'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=price_range_counts.index, y=price_range_counts.values, palette="viridis")
plt.xlabel('Price Range')
plt.ylabel('Number of Restaurants')
plt.title('Distribution of Restaurants by Price Range')
plt.xticks(rotation=0)
plt.show()


# ### CALCULATING THE AVERAGE RATING FOR EACH PRICE RANGE.

# In[24]:


avg_ratings_price_range = dataset.groupby('Price range')['Aggregate rating'].mean().reset_index()
avg_ratings_price_range


# ### VISUALIZING THE AVERAGE RATING FOR EACH PRICE RANGE.

# In[25]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Price range', y='Aggregate rating', data=avg_ratings_price_range, palette='viridis')

plt.xlabel('Price Range')
plt.ylabel('Average Aggregate Rating')
plt.title('Average Ratings for Each Price Range')
plt.ylim(0, 5)
plt.show()


# ### IDENTIFYING THE COLOR THAT REPRESENTS THE HIGHEST AVERAGE RATING AMONG DIFFERENT PRICE RANGES.

# In[28]:


avg_ratings_price_range = dataset.groupby('Price range')['Aggregate rating'].mean().reset_index()

max_avg_rating = avg_ratings_price_range['Aggregate rating'].max()
max_price_range = avg_ratings_price_range[avg_ratings_price_range['Aggregate rating'] == max_avg_rating]['Price range'].values[0]

plt.figure(figsize=(10, 6))
bars = sns.barplot(x='Price range', y='Aggregate rating', data=avg_ratings_price_range, palette='viridis')

for bar, rating, price_range in zip(bars.patches, avg_ratings_price_range['Aggregate rating'], avg_ratings_price_range['Price range']):
    if price_range == max_price_range:
        bar.set_color('red')
        plt.annotate(f'Highest Avg Rating\nPrice Range: {price_range}', 
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                     xytext=(0, 3),  
                     textcoords='offset points', 
                     ha='center', va='bottom', 
                     color='black', weight='bold')

plt.xlabel('Price Range')
plt.ylabel('Average Aggregate Rating')
plt.title('Average Ratings for Each Price Range')
plt.ylim(0, 5)

plt.show()


# # TASK 3 : FEATURE ENGINEERING

# ### EXTRACTING ADDITIONAL FEATURES FROM THE EXISTING COLUMNS, SUCH AS THE LENGTH OF THE RESTAURANT NAME OR ADDRESS.

# ### LENGTH OF THE RESTAURANTS NAME

# In[34]:


dataset['Restaurant Name Length'] = dataset['Restaurant Name'].apply(len)
dataset['Address Length'] = dataset['Address'].apply(len)

print(dataset[['Restaurant Name', 'Restaurant Name Length']].head(15))


# ### LENGTH OF THE RESTAURANTS ADDRESS

# In[38]:


print(dataset[['Address', 'Address Length']].head(15))


# ### EXTRACTING SOME MORE FEATURES FROM THE EXISTING COLUMNS

# In[6]:


dataset['Restaurant Name Word Count'] = dataset['Restaurant Name'].apply(lambda x: len(x.split()))
dataset['Address Word Count'] = dataset['Address'].apply(lambda x: len(x.split()))
dataset['Restaurant Name Special Char'] = dataset['Restaurant Name'].apply(lambda x: any(char in x for char in '&-'))
dataset['Address Contains Street'] = dataset['Address'].apply(lambda x: 'Street' in x or 'St.' in x or 'St' in x)
dataset['Cuisine Count'] = dataset['Cuisines'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

print(dataset[['Restaurant Name Word Count', 'Restaurant Name Special Char']].head(15))


# In[5]:


print(dataset[['Address Word Count', 'Address Contains Street']].head(15))


# In[7]:


print(dataset[['Cuisines', 'Cuisine Count']].head(15))


# ### CREATING NEW FEATURES LIKE 'HAS TABLE BOOKING' OR  'HAS ONLINE DELIVERY' BY ENCODING CATEGORICAL VARIABLES.

# ### ENCODING 'HAS TABLE BOOKING'

# In[10]:


dataset['Has Table Booking'] = dataset['Has Table booking'].map({'Yes': 1, 'No': 0})
dataset['Has Online Delivery'] = dataset['Has Online delivery'].map({'Yes': 1, 'No': 0})

print(dataset[['Has Table booking', 'Has Table Booking']].head(15))


# ### ENCODING 'HAS ONLINE DELIVERY'

# In[9]:


dataset['Has Online Delivery'] = dataset['Has Online delivery'].map({'Yes': 1, 'No': 0})
print(dataset[['Has Online delivery', 'Has Online Delivery']].head(15))

