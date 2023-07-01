# E-Commerce Data 

In this project, the aim is to predict whether the next customer will make a reorder or not for Hunter's E-grocery, a prominent French brand in the e-grocery and lifestyle products industry. By leveraging the power of big data analytics, I will explore customer behavior patterns and develop a predictive model to optimize marketing campaigns and reduce expenses.

I will be working with a comprehensive dataset that captures various aspects of customer orders. The dataset includes information such as order IDs, user IDs, order numbers, the day of the week the order was placed, the hour of the day the order was placed, the number of days since the user's previous order, product IDs, the order in which products were added to the cart, department IDs, and more. Most importantly, the dataset provides a binary indicator, the "reordered" column, which tells whether a particular product has been ordered by the user in the past.

## Dataset

The dataset used for this project contains the following variables:

ECommerce_consumer_behaviour.csv dataset (2019501 data)

https://www.kaggle.com/datasets/hunter0007/ecommerce-dataset-for-predictive-marketing-2023

- `order_id`: numeric - Unique identifier for each order
- `user_id`: - numeric - Unique identifier for each user who placed an order
- `order_number`: - numeric - The sequence number for each order placed by the user
- `order_dow`: - numeric - The day of the week (0-6, where 0 is Sunday) on which the order was placed
- `order_hour_of_day`: - numeric - The hour of the day (0-23) at which the order was placed
- `days_since_prior_order`: - numeric - Number of days since the user's previous order
- `product_id`: - numeric - Unique identifier for each product
- `add_to_cart_order`: - numeric - The order in which the product was added to the cart
- `department_id`: - numeric - Unique identifier for each department
- `department`: - string - Unique identifier for each department
- `product_name`: - string - The name of the product
- `reordered(target)`: - numeric - Binary indicator for whether the product has been ordered by the user in the past

## Model Building

To achieve the objective, it is followed a systematic approach that involves several stages. It is started by preparing the data, cleaning, and organizing it for analysis. Next, perform exploratory data analysis (EDA) to gain insights into the dataset and identify any patterns or trends. Once I have a thorough understanding of the data, I will proceed to train and evaluate predictive models using machine learning techniques such as:



 Logistic Regression

 Random Forest

 Gradient Boosting Classifier 
 
 

Additionally, Hypothesis testing by using Chi-square is applied to the features to understand their importance for our dependent column.

The models are trained to predict the probability of customer reorders based on the available features. Finally, the model's performance will be evaluated, and draw meaningful conclusions to guide Hunter's E-grocery in optimizing its marketing campaigns.
Following this approach is aimed to provide valuable insights into customer reorder behavior and enable Hunter's E-grocery to make data-driven decisions, enhancing customer satisfaction and driving business growth.

## Usage

- Databricks

## Results 

Gradient Boosting Classifier is selected as the best model. 

