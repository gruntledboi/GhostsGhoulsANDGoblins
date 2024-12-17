library(tidymodels)
library(keras)
library(remotes)
library(bonsai)
library(lightgbm)
library(vroom)
library(dbarts)
library(lme4)
library(parsnip)
library(naivebayes)
library(themis)
library(discrim)

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

trainData$type <- as.factor(trainData$type)

my_recipe <- recipe(formula = type ~ ., data = trainData) %>%
  step_mutate(id, features = id) |> 
  step_mutate(color = as.factor(color))


## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
tuning_grid <- grid_regular(finalize(Laplace(), trainData),
                            smoothness(),
                            levels = 9)

## Set up K-fold CV
folds <- vfold_cv(trainData, v = 10, repeats = 1)

CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "accuracy")


## Predict
final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)


ggg_naive_bayes_predictions <- 
  predict(final_wf,
          new_data = testData,
          type = "class") |> 
  bind_cols(testData) |> 
  rename(type = .pred_class) |> 
  select(id, type)

vroom_write(x = ggg_naive_bayes_predictions, 
            file="./GGGNaiveBayesPreds.csv", delim=",")
