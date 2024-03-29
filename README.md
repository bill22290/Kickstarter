## Predicting Kickstarter Project Success

For this project I will be analyzing a Kickstarter dataset in an attempt to identify if any feature variable attributes in the data are predictive of Kickstarter project success or failure.  I will be working with Machine Learning classification algorithms in R, specifically a random forest  model and a decision tree model (rpart), throughout the course of the exercise. 

The dataset to  be used for this project was originally posted on Kaggle retrieved from: https://www.kaggle.com/kemical/kickstarter-projects

### Overview

-The original dataset has over 320,000 Kickstarter project entries from 2009-2016

-Feature Variables to be examined include: fundraising campaign category, currency, deadline date, launch date, amount pledged and number of backers

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

As we can see from the Data Visualization pieces produced above, the distribution for the number of backers per fund raising project as well as the distribution for U.S. Dollars pledged per project are skewed to the left.  This tells us that the majority of Kickstarter fundraising projects are funded at relatively low U.S. dollar amounts involving many donors. 

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
I need to filter out entries that have irrelevant State variable entries (i.e. any State variable not equal to "failed" or "successful") . I also want to create another dataframe in my environment to begin parsing the data into a structure that will work for a random forest model and decision tree. For example, the X - X.3 columns in the original dataframe appear to be metadata that is unnecessary for the purpose of my project, so I will begin dropping those columns as I start to manipulate the data into a new frame.
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
I am now ready to build my Random Forest model.  I have created a new data frame only including the variables that I want to use for the model:
```
#Note that I have normalized the usd.pledged variable as well as the date_diff variable in order to make sure the numeric columns don't distort the results of the model
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
>set.seed(123)
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
The Variable Importance plot shows us that the amount of U.S. dollars pledged is by far the most important variable in the model for predicting the classification of a Kickstarter project as either a success or failure. 

### Decision Tree - Rpart
I want to compare the Random Forest model that I just built, "model1", to a decision tree built from the Rpart package in R using the same dataframe. 
```
#Important to remember for this model to specify method = "class" since I am using classification for this project
> mytree <- rpart(kickstarter_time_test$state ~ kickstarter_time_test$main_category+ kickstarter_time_test$currency + kickstarter_time_test$usd.pledged + kickstarter_time_test$date_diff, data = kickstarter_time_test, method = "class", parms = list(split = 'information'), minsplit = 2, minbucket = 1)
> rpart.plot(mytree)
```
![](https://github.com/bill22290/Kickstarter/blob/master/images/Rpart_mytree.png)

Looking at the decision tree we can tell that the most important feature is the amount of USD pledged and the point of delineation is at -1.3 normalized USD pledged. The second most important feature is the fundraiser main category where projects with categories = Crafts, Design, Fashion, Food, Games, Journalism, Photography, Publishing and Technology had a better chance of being successful compared to the rest of the sample.

Since I normalized the USD pledged column, I want to back into a dollar figure that gives me an idea of what is the point of demarcation where fundraising projects below a certain threshold are much more likely to fail. 
```
>lowUSD <- filter(kickstarter_time_test, kickstarter_time_test$usd.pledged < -1.3)
> str(lowUSD)
'data.frame':	38357 obs. of  5 variables:
 $ main_category: Factor w/ 15 levels "Art","Comics",..: 13 13 3 9 3 7 13 6 14 6 ...
 $ currency     : Factor w/ 13 levels "AUD","CAD","CHF",..: 6 2 13 13 6 13 13 5 2 13 ...
 $ state        : Factor w/ 2 levels "failed","successful": 1 1 1 1 1 1 1 1 1 1 ...
 $ usd.pledged  : num  -1.32 -1.32 -1.32 -1.32 -1.32 ...
 $ date_diff    : num  1.925 -0.326 -0.326 0.838 -0.326 ...
 ```
If I am narrowing in on the observations from the kickstarter_time_test dataframe which have USD pledged less than -1.3 on a normalized basis, that means I am focusing on the lower 13.6 percentile of the population (38,357 observations/281,302 observations).
```
>quantile(kickstarter_time$usd.pledged, c(.136))
  13.6% 
457.936
```
If I was working at Kickstarter and wanted to increase the overall fundraising project success rate, I would be targeting projects right around and below the $457 USD pledged threshold as those projects are most likely to fail and are in need of fundraising assistance based upon the rpart "mytree" model.

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
The Rpart model has a better Recall score, however the Random Forest model has better accuracy, precision, F1 value and Kappa value. At first glance, the Random Forest model looks like the better of the two. However, it is important to rememeber the costs associated with False Positives and Negatives. Precision is the better statistic to consider when comparing models where the cost of a False Positive is high and Recall is the more pertinent figure to look at when the cost of False Negative is high.

Notice that for my classification models, the Possitive class = failed which might be counterintuitive. If a project that is going to fail (Actual Positive) goes through the model and is predicted as a success (Predicted Negative) that is a worse outcome than a project which is going to succeed (Actual Negative) is predicted to fail (Predicted Positive).  Recall is the better metric to look at for my models because there is a higher cost associated with False Negatives.  Consider a false negative, where a project that is going to fail, is predicted to succeed. This project would not get the preemptive assistance that Kickstarter could offer for projects that are predicted to fail. 

If I check the accuracy of "model1" predictions against the valid.kick data I get similar statistics as when running "model1" with the train.kick data:
```
> testP <- predict(model1, valid.kick)
> caret::confusionMatrix(testP, valid.kick$state, mode = "prec_recall")
Confusion Matrix and Statistics

            Reference
Prediction   failed successful
  failed      40196      12824
  successful  10495      20876
                                          
               Accuracy : 0.7237          
                 95% CI : (0.7206, 0.7267)
    No Information Rate : 0.6007          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.4173          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
              Precision : 0.7581          
                 Recall : 0.7930          
                     F1 : 0.7752          
             Prevalence : 0.6007          
         Detection Rate : 0.4763          
   Detection Prevalence : 0.6283          
      Balanced Accuracy : 0.7062          
                                          
       'Positive' Class : failed   
```
The Accuracy, Kappa, Precision, Recall and F1 figures are identical (only differing by decimals) when running the valid.kick data with "model1" compared to using the training.data set with "model1".  

## Cross Validation
![](https://github.com/bill22290/Kickstarter/blob/master/images/RF_CV.PNG)
```
> with(rfk, plot(rfk$n.var, rfk$error.cv, log = "x", type = "o", lwd=2))
```
![](https://github.com/bill22290/Kickstarter/blob/master/images/RFCV5_Plot.png)

I expected a linear relationship between the C.V. error rate and the number of variables. I thought it was strange that this first RFCV plot has a C.V. error rate of approx. 31% with one variable, spiked to 32% for two variables and then had the lowest error rate of approx. 30% for 4 variables. 

The rfcv help page in R offers an example to replicate the cross validation process to get a larger sample:
![](https://github.com/bill22290/Kickstarter/blob/master/images/RFCV5_Replicate.PNG)
After replicating the rfcv() function five times and then plotting the results, we can see a more clear pattern that as the number of variables increase, the C.V. error rate decreases. Perhaps the reason for a spike in the error rate with two variables was a result of the most important predictive variable, USD pledged, being the sole variable when n.var = 1 and thus having a disproportionate impact on the C.V. error rate when n.var = 1.

(Note that I set the number of cv.fold = 5, for cv.fold > 5 my machine ran into memory constraints)
## Rpart Printcp()
```
#Set xval = 5 in rpart.control() for number of folds to compare to the first random forest cross validation conducted which had cv.fold = 5
>rpart::rpart.control(xval = 5)
> rpart::printcp(mytree)

Classification tree:
rpart(formula = kickstarter_time_test$state ~ kickstarter_time_test$main_category + 
    kickstarter_time_test$currency + kickstarter_time_test$usd.pledged + 
    kickstarter_time_test$date_diff, data = kickstarter_time_test, 
    method = "class", parms = list(split = "information"), minsplit = 2, 
    minbucket = 1)

Variables actually used in tree construction:
[1] kickstarter_time_test$date_diff     kickstarter_time_test$main_category kickstarter_time_test$usd.pledged  

Root node error: 113081/281302 = 0.40199

n= 281302 

        CP nsplit rel error  xerror      xstd
1 0.073054      0   1.00000 1.00000 0.0022996
2 0.012314      2   0.85389 0.85203 0.0022258
3 0.010000      4   0.82926 0.82864 0.0022106
```
When running the printcp() function on rpart classification trees, the cross-validated error rate equals the Root node error * the xerror rate which in this case is (.40199) * (.82864) = approx. 33% when nsplit = 4 which is a higher error rate than the cv.error = 30% when n.var = 4 in "model1". When running cross-validation, the rpart decision tree model "mytree" has a higher cross-validation error rate with four variables than the Random Forest Model "model1".

## Conclusion

The goal of this project was to try and determine if there was any predictive relationship between a feature variable attribute in the Kickstarter dataset and the project state (success or failure) variable.  Looking at the model statistics, "model1" overall had better metrics than "mytree", however based upon my particular problem, Recall actually is the most important statistic so "mytree" would be more useful in this analysis. That being said, neither are exceptionally accurate. Many analysts look at a Kappa value of .40 as a minimum acceptable amount and only the random forest model, "model1", had a Kappa > .40. However, even if the accuracy and Kappa numbers in particular were a little underwhelming, the exercise still uncovered some useful information in regards to Kickstarter projects. 

Both models confirmed that total USD pledged was the most important variable in terms of predicting Kickstarter project success or failure. The decision tree model built in rpart indicated that projects at or lower than the 13.6 percentile for USD pledged, or approx. $457, are much more likely to fail. I also learned that Kickstarter project category was the second most important variable in the rpart decision tree for determining project success, followed by the amount of time that the project was open as the third most important variable. 

When I interpret the results of this analysis, I consider the objective of Kickstarter as an organization, which is to help bring creative projects to life.  I approached this topic from the standpoint of trying to predict overall fundraising success or failure rate. If Kickstarter can accurately predict a project's success or failure based upon feature attributes, they would be in a much better position to increase the overall funding success rate which in turn helps acheive their goal.  Perhaps Kickstarter is already aware that projects below a few hundred dollars are much more likely to fail.  One of the great benefits that Fintech companies have is the ability to deploy economies of scale.  They can leverage their technology to provide best business practices to customers at higher volume with relatively lower marginal cost. Running through an exercise similar to the one I just conducted would cost litte to Kickstarter. Kickstarter could preemptively target projects below a certain USD amount threshold ($457 in my model) and immediately offer additional assistance that will help achieve the fundraising goal for those lower dollar projects. This type of approach would move the organization closer to achieving their stated goal.

## Sources

https://towardsdatascience.com/
https://github.com/
https://stackoverflow.com/
https://www.rdocumentation.org/
https://www.r-bloggers.com/how-to-implement-random-forests-in-r/
https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
