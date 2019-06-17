Course Project in the Coursera Course Practical Machine Learning
----------------------------------------------------------------

The goal of this project is to predict the manner in which the participants in the study did the exercise (on a rating from A-E). This involves describing how the final model was built; how cross-validation was used; what I expected the out of sample error to be and why I made the choices I did.

The chunk below loads the required packages and the training dataset. Note that all na-characters in the dataset are imported as `NA` with the `na =`-command.

``` r
rm(list = ls())
library(tidyverse); library(caret); library(magrittr); library(parallel); library(doParallel)
training <- read_csv("Datasets/pml-training.csv", na = c("", "-", "NA", "#DIV/0!", "<NA>"))
```

    ## Warning: Missing column names filled in: 'X1' [1]

First of all, the dataset requires some cleaning. There are a lot of variables with no values and only NA's. I remove all variables with more than 95 % missing observations. After this, there are three missing values which I replace with the median values (not the mean as there are outliers in the data) of the same variables.

``` r
# remove all features with more than 95 pct. of observations missing - leaves us with 59 features
training <- training[, -c(which(apply(training, 2, function(x) sum(is.na(x))) > 19000))] 

# three missing values - magnet_forearm_y and z AND magnet_dumbbell_z
which(apply(training, 2, function(x) sum(is.na(x))) > 0)
```

    ## named integer(0)

``` r
# replace the missing values with the median values for that variable (imputation)
training[is.na(training$magnet_forearm_y), "magnet_forearm_y"] <- median(training$magnet_forearm_y, na.rm=T)
training[is.na(training$magnet_forearm_z), "magnet_forearm_z"] <- median(training$magnet_forearm_z, na.rm=T)
training[is.na(training$magnet_dumbbell_z), "magnet_dumbbell_z"] <- median(training$magnet_dumbbell_z, na.rm=T)
```

I also remove all the variables related to the user and the date/time, as this should not be used in a classification model. One could expect that the name of the user would be a good predictor but this wouldn't help the model predict out of sample.

``` r
# remove unimportant variables such as user data and time stamps
# these variables don't apply to unknow data and are thus irrelevant for a prediction model
training$user_name <- NULL
training$raw_timestamp_part_1 <- NULL
training$raw_timestamp_part_2 <- NULL
training$cvtd_timestamp <- NULL
training$new_window <- NULL
training$num_window <- NULL
```

Furthermore, one observation has extreme values on `gyros_dumbbell_x`and `gyros_dumbbell_z`that seem to be errors in the dataset. I remove this observation so it doesn't interfere with the model.

``` r
# preprocessing
# don't print here: summary(training)

# extreme values in _gyros_?
max(training$gyros_dumbbell_x); min(training$gyros_dumbbell_x)
```

    ## [1] 2.22

    ## [1] -204

``` r
max(training$gyros_dumbbell_z); min(training$gyros_dumbbell_z)
```

    ## [1] 317

    ## [1] -2.38

``` r
# don't print here: training %>% filter(gyros_dumbbell_x < -10) %>% as.data.frame()

# removing this observation - seems like an outlier / or a faulty measurement
training <- training[!training$X1 == 5373, ] 
```

I decided to build the model as a random forest (`method = "rf"` in the `caret`-package). I use cross-validation (7-folds) to avoid overfitting the model. I also use the `parallel`-package to speed up the estimation of the model.

``` r
# RF with cross-validation
train_control <- trainControl('cv', 7)

cl <- makePSOCKcluster(4)
registerDoParallel(cl)

fit1.1 <- caret::train(classe ~., data = training[-1], method = "rf", trControl = train_control) # 53 predictors
fit1.1 
```

    ## Random Forest 
    ## 
    ## 19621 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (7 fold) 
    ## Summary of sample sizes: 16819, 16818, 16818, 16819, 16817, 16817, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9950564  0.9937466
    ##   27    0.9948014  0.9934238
    ##   52    0.9901124  0.9874916
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

``` r
# don't print here: Sys.time()
stopCluster(cl)
```

As the accuracy in the initial model is higher than 99 pct. there seems to be no reason to develop the model any further. As I have used cross-validation, I also expect the out-of-sample error to be fairly low.

``` r
# classifying the test-data using rf-model
testdata <- read_csv("Datasets/pml-testing.csv", na = c("", "-", "NA", "#DIV/0!", "<NA>"))
```

    ## Warning: Missing column names filled in: 'X1' [1]

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_logical(),
    ##   X1 = col_double(),
    ##   user_name = col_character(),
    ##   raw_timestamp_part_1 = col_double(),
    ##   raw_timestamp_part_2 = col_double(),
    ##   cvtd_timestamp = col_character(),
    ##   new_window = col_character(),
    ##   num_window = col_double(),
    ##   roll_belt = col_double(),
    ##   pitch_belt = col_double(),
    ##   yaw_belt = col_double(),
    ##   total_accel_belt = col_double(),
    ##   gyros_belt_x = col_double(),
    ##   gyros_belt_y = col_double(),
    ##   gyros_belt_z = col_double(),
    ##   accel_belt_x = col_double(),
    ##   accel_belt_y = col_double(),
    ##   accel_belt_z = col_double(),
    ##   magnet_belt_x = col_double(),
    ##   magnet_belt_y = col_double(),
    ##   magnet_belt_z = col_double()
    ##   # ... with 40 more columns
    ## )

    ## See spec(...) for full column specifications.

``` r
# same set of predictors as in training
testdata <- testdata[, intersect(names(testdata), names(training))]

# check for any NA's
sum(is.na(testdata))
```

    ## [1] 0

``` r
# predict
testdata_predicted <- predict(fit1.1, newdata = testdata)

testdata$predicted_classe <- as.character(testdata_predicted)

# Test classification
test_fit <- data.frame(true_val = training$classe, predicted_val = predict(fit1.1), stringsAsFactors = F)
test_fit %<>%
  mutate(correct = as.numeric(true_val == predicted_val)) # 100 %

# for the quiz
testdata %>% select(X1, predicted_classe)
```

    ## # A tibble: 20 x 2
    ##       X1 predicted_classe
    ##    <dbl> <chr>           
    ##  1     1 B               
    ##  2     2 A               
    ##  3     3 B               
    ##  4     4 A               
    ##  5     5 A               
    ##  6     6 E               
    ##  7     7 D               
    ##  8     8 B               
    ##  9     9 A               
    ## 10    10 A               
    ## 11    11 B               
    ## 12    12 C               
    ## 13    13 B               
    ## 14    14 A               
    ## 15    15 E               
    ## 16    16 E               
    ## 17    17 A               
    ## 18    18 B               
    ## 19    19 B               
    ## 20    20 B
