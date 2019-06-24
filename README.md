## Predicting Kickstarter Success

For this project I will be analyzing a Kickstarter dataset in an attempt to identify if any feature variable attributes in the dataframe are predictive of Kickstarter project success or failure.  I will be working with Machine Learning algorithms in R, specifically a random forest and a decision tree model, throughout the course of the exercise. 

The dataset to  be used for this project was originally posted on Kaggle retrieved from: https://www.kaggle.com/kemical/kickstarter-projects

### Overview

-The original dataset has over 320,000 Kickstarter project entries from 2009-2016

-Feature Variables to be examined include: fundraising campaign category, currency, deadline date, fundraising goal, launch date, amount pledged and number of backers

## Exploratory Data Analysis
```
> kickstarter <- read.csv("ks-projects.csv", header = TRUE)
> str(kickstarter)
'data.frame':	323750 obs. of  17 variables:
 $ ID           : int  1000002330 1000004038 1000007540 1000011046 1000014025 1000023410 1000030581 1000034518 100004195 100004721 ...
 $ name         : Factor w/ 321577 levels "'' Album''Eyes to Eyes''of Kilimandjaro' '",..: 284698 312444 295356 66665 176323 250667 59831 242722 248308 191218 ...
 $ category     : Factor w/ 771 levels "","' Garland Wright and the American Theater",..: 717 700 697 660 733 664 646 722 643 702 ...
 $ main_category: Factor w/ 120 levels " 50 Years in the Making",..: 93 49 72 49 52 52 52 35 49 93 ...
 $ currency     : Factor w/ 37 levels " Be active!",..: 20 37 37 37 37 37 37 37 37 8 ...
 $ deadline     : Factor w/ 275294 levels " Esoteric","1/1/2010 0:00",..: 40066 98726 134086 243225 129107 74869 111738 169033 226443 39750 ...
 $ goal         : Factor w/ 8188 levels "0.01","0.15",..: 91 5166 5578 2346 5580 91 3172 860 6496 3170 ...
 $ launched     : Factor w/ 295793 levels "1/1/1970 1:00",..: 247839 1821 121009 240356 106417 74046 91457 154213 219941 295036 ...
 $ pledged      : Factor w/ 55598 levels "0","1","1.01",..: 1 20162 2 6991 40856 5377 37321 51505 45238 1 ...
 $ state        : Factor w/ 410 levels "0","1","1/28/2015 13:53",..: 406 406 406 405 408 408 406 405 405 406 ...
 $ backers      : Factor w/ 3592 levels "0","1","10","100",..: 1 1851 2 520 1368 748 2304 2828 2409 1 ...
 $ country      : Factor w/ 162 levels "0","1","10","107",..: 149 162 162 162 162 162 162 162 162 142 ...
 $ usd.pledged  : Factor w/ 94377 levels "","0","0.566314",..: 2 34611 565 12318 69164 9880 63347 87104 76210 2 ...
 $ X            : Factor w/ 417 levels "","0","0.75115847",..: 1 1 1 1 1 1 1 1 1 1 ...
 $ X.1          : Factor w/ 9 levels "","0","1","128534.5877",..: 1 1 1 1 1 1 1 1 1 1 ...
 $ X.2          : Factor w/ 4 levels "","0","9854",..: 1 1 1 1 1 1 1 1 1 1 ...
 $ X.3          : int  NA NA NA NA NA NA NA NA NA NA ...
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

### Data Cleaning
I am only interested in Kickstarter projects that were either cataloged as successes or failures, I am not interested in projects that had incomplete data entry.  Notice when running the str() function on on the dataset, the state variable was inputted as a factor with 410 levels.  
```
> summary(kickstarter)
state         
failed     :168221
successful :113081
canceled   : 32354
live       :  4428
undefined  :  3555
suspended  :  1479
(other)    :   632
```
