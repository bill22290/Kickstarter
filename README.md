## Predicting Kickstarter Success

For this project I will be analyzing a Kickstarter dataset in an attempt to identify if any feature variable attributes in the dataframe are predictive of Kickstarter project success or failure.  I will be working with Machine Learning algorithms in R, specifically a random forest classification model and a decision tree model, throughout the course of the exercise. 

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

As we can see from the Data Visualization pieces produced above, the distribution for the number of backers per fund raising project as well as the distribution for U.S. Dollars pledged per project are skewed to the left.  This tells us that the majority of kickstarter fundraising projects are funded at relatively low U.S. dollar amounts involving many donors. 

### Data Cleaning
I am only interested in Kickstarter projects that were either cataloged as successes or failures, I am not interested in projects that had incomplete project state data entry.  Notice when running the str() function on the dataset, the state variable was inputted as a factor with 410 levels. If we use the summary() function and only look at the state variable, it appears that entries with a state variable other than "failed" or "successful" were inputted at different factor levels.  
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
I need to filter out entries that have irrelevant State variable entries (i.e. any State variable not equal to "failed" or "successful") . I also want to create another dataframe in my environment to begin parsing the data into a structure that will work for a random forest model and decision tree. For example, the X - X.3 columns in the original dataframe appear to be metadata that is unnecessary for the purose of my project, so I will begin dropping those columns as I start to maniplate the data into a new frame.
```
> kickstarter2 <- kickstarter[,-17]
> kickstarter2 <- kickstarter2[,-2]
str(kickstarter2)
> str(kickstarter2)
'data.frame':	323750 obs. of  15 variables
```
The dplyr library in R will be needed in order to filter variables:
```
> library(dplyr)
> kickstarter_test <- filter(kickstarter2, kickstarter2$state == "successful" | kickstarter2$state == "failed")
> str(kickstarter_test)
'data.frame':	281302 obs. of  15 variables:
 $ ID           : int  1000002330 1000004038 1000007540 1000014025 1000023410 1000030581 100004721 100005484 1000055792 1000056157 ...
 $ category     : Factor w/ 771 levels "","' Garland Wright and the American Theater",..: 717 700 697 733 664 646 702 680 636 668 ...
 $ main_category: Factor w/ 120 levels " 50 Years in the Making",..: 93 49 72 52 52 52 93 72 33 54 ...
 $ currency     : Factor w/ 37 levels " Be active!",..: 20 37 37 37 37 37 8 37 37 37 ...
 $ deadline     : Factor w/ 275294 levels " Esoteric","1/1/2010 0:00",..: 40066 98726 134086 129107 74869 111738 39750 150721 25617 118634 ...
 $ goal         : Factor w/ 8188 levels "0.01","0.15",..: 91 5166 5578 5580 91 3172 3170 859 5578 2498 ...
 $ launched     : Factor w/ 295793 levels "1/1/1970 1:00",..: 247839 1821 121009 106417 74046 91457 295036 139509 280871 113951 ...
 $ pledged      : Factor w/ 55598 levels "0","1","1.01",..: 1 20162 2 40856 5377 37321 1 6763 1 1 ...
 $ state        : Factor w/ 410 levels "0","1","1/28/2015 13:53",..: 406 406 406 408 408 406 406 408 406 406 ...
 ```
 Notice how the total number of observations for the new data.frame has dropped to 281,302 observations compared to 323,750 observations for the original dataframe. I now need to drop the unused factor levels in my state variable since I removed all entries that do not have a kickstarter state variable equalling "successful" or "failed":
 ```
 > kickstarter_test$state <- droplevels(kickstarter_test$state)
> str(kickstarter_test$state)
 Factor w/ 2 levels "failed","successful": 1 1 1 2 2 1 1 2 1 1 ...
 ```
### Feature Engineering
The original dataset on Kaggle only had columns for a project launch date and deadline date. I want to engineer a feature variable that will tell me how long each project was active:
```
> date_diff <- as.numeric(as.Date(as.character(kickstarter_test$deadline), format = "%m/%d/%Y")-as.Date(as.character(kickstarter_test$launched), format = "%m/%d/%Y"))
> str(date_diff)
 num [1:281302] 59 45 30 35 20 45 30 30 30 45 ...
 #I now want to bind the newly created "date_diff" as a new feature variable to the dataframe that I am working with
 >kickstarter_time <- cbind(kickstarter_test, date_diff)
```

### Random Forest Model
I am now ready to build my Random Forest model.  I have created a new data frame only including the variables that I want to include for the model:
```
#Note that I have normalized the usd.pledged variable as well as the date_diff variable as those two are numeric
> str(kickstarter_time_test)
'data.frame':	281302 obs. of  5 variables:
 $ main_category: Factor w/ 15 levels "Art","Comics",..: 13 7 11 8 8 8 13 11 3 9 ...
 $ currency     : Factor w/ 13 levels "AUD","CAD","CHF",..: 6 13 13 13 13 13 2 13 13 13 ...
 $ state        : Factor w/ 2 levels "failed","successful": 1 1 1 2 2 1 1 2 1 1 ...
 $ usd.pledged  : num  -1.318 -0.159 -1.299 0.999 -0.987 ...
 $ date_diff    : num  1.9246 0.838 -0.3261 0.0619 -1.1023 ...
```
Next I will organize the dataframe into a training (70% of the data) and valid data group (30% of the data):
```
>train <- sample(nrow(kickstarter_time_test), 0.7*nrow(kickstarter_time_test), replace = FALSE)
>train.kick <- kickstarter_time_test[train,]
valid.kick <- kickstarter_time_test[-train,]
#Once the data has been partitioned, I can now set up my model:
model1 <- randomForest(train.kick$state ~., data = train.kick, ntree = 500, mtry = 3, importance = TRUE)
> model1

Call:
 randomForest(formula = train.kick$state ~ ., data = train.kick,      ntree = 500, mtry = 3, importance = TRUE) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 3

        OOB estimate of  error rate: 27.66%
Confusion matrix:
           failed successful class.error
failed      93675      23855   0.2029695
successful  30619      48762   0.3857220
> varImpPlot(model1)
```
![](https://github.com/bill22290/Kickstarter/blob/master/images/Model1_Var_Imp.png)
The Variable Importance plot shows us that the amount of U.S. dollars pledged is the most important variable in the model. 

### Decision Tree - Rpart
I want to compare the Random Forest model that I just built, model1, to a decision tree built from the Rpart package in R using the same dataframe. 
```
#Important to remember for this model to specify method = "class" since I am using classification for this project
> mytree <- rpart(kickstarter_time_test$state ~ kickstarter_time_test$main_category+ kickstarter_time_test$currency + kickstarter_time_test$usd.pledged + kickstarter_time_test$date_diff, data = kickstarter_time_test, method = "class", parms = list(split = 'information'), minsplit = 2, minbucket = 1)
> rpart.plot(mytree)
```
![](https://github.com/bill22290/Kickstarter/blob/master/images/Rpart_mytree.png)

## Comparing Models
The first split in a decision tree will be the most important feature. The VarImPlot() for the Random Forest model and the rpart.plot() for the decision tree model both indicate that the most important variable for predicting Kickstarter project success or failure is the amount of U.S. pledged.  

## Confusion Matrix
```
> library(caret)
#I am interested in the Precision and Recall statistics in order to compare model accuracy, so mode = "prec_recall"
> caret::confusionMatrix(model1$predicted, model1$y, mode = "prec_recall")
#To create a confusion Matrix for my decision tree I need to create a prediction table using my decision tree model to compare the model's accuracy with respect to the actual project state classifications (success or failure).
> P <- predict(mytree, type = "class")
> table(P)
P
    failed successful 
    214717      66585 
#Remember to set the y variable in the decision tree model to a factor for confusionMatrix
> mytree$y <- as.factor(mytree$y)
#Defining levels for the y variable in the mytree model
> levels(mytree$y) <- c('failed', 'successful')
> table(mytree$y)

    failed successful 
    168221     113081 
>> caret::confusionMatrix(P, mytree$y, mode = "prec_recall")
```
![](https://github.com/bill22290/Kickstarter/blob/master/images/RF_RPart.PNG)





