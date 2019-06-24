## Predicting Kickstarter Success

For this project I will be analyzing a Kickstarter dataset in an attempt to identify if any feature variable attributes in the dataframe are predictive of Kickstarter project success or failure.  I will be working with Machine Learning algorithms in R, specifically a random forest and a decision tree model, throughout the course of the exercise. 

The dataset to  be used for this project was originally posted on Kaggle retrieved from: https://www.kaggle.com/kemical/kickstarter-projects

### Overview

-The original dataset has over 320,000 Kickstarter project entries from 2009-2016

-Feature Variables to be examined include: fundraising campaign category, currency, deadline date, fundraising goal, launch date, amount pledged and number of backers

### Exploratory Data Analysis

```markdown

> kickstarter <- read.csv("ks-projects.csv", header = TRUE)
> boxplot(as.numeric(kickstarter$usd.pledged), main = "USD Pledged")

![USDP](https://raw.githubusercontent.com/bill22290/Kickstarter/blob/master/USDPledged_box.png)

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/bill22290/Kickstarter/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
