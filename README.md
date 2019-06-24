## Predicting Kickstarter Success

For this project I will be analyzing a Kickstarter dataset in an attempt to identify if any feature variable attributes in the dataframe are predictive of Kickstarter project success or failure.  I will be working with Machine Learning algorithms in R, specifically a random forest and a decision tree model, throughout the course of the exercise. 

The dataset to  be used for this project was originally posted on Kaggle retrieved from: https://www.kaggle.com/kemical/kickstarter-projects

### Overview

-The original dataset has over 320,000 Kickstarter project entries from 2009-2016

-Feature Variables to be examined include: fundraising campaign category, currency, deadline date, fundraising goal, launch date, amount pledged and number of backers

## Exploratory Data Analysis
```
> kickstarter <- read.csv("ks-projects.csv", header = TRUE)
> boxplot(as.numeric(kickstarter$usd.pledged), main = "USD Pledged")
```
![](https://github.com/bill22290/Kickstarter/blob/master/images/USDPledged_box.png)
```
> hist(as.numeric(kickstarter$usd.pledged))
```
![](https://github.com/bill22290/Kickstarter/blob/master/images/USDPledged_hist.png)
```
> hist(as.numeric(kickstarter$backers))
```
![](https://github.com/bill22290/Kickstarter/blob/master/images/Backers_hist.png)

As we can see from the Data Visualization pieces produced above, the distribution for the number of backers per fund raising project as well as the distribution for U.S. Dollars pledged are skewed to the left. 
