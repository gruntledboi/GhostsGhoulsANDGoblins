library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(lme4)
library(parsnip)
library(discrim)
library(naivebayes)
library(themis)

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

trainData$ACTION <- as.factor(trainData$ACTION)

my_recipe <- recipe(type ~ ., data = trainData) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |>  # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) |>  # combines categorical values that occur
  #step_dummy(all_nominal_predictors()) |>  # dummy variable encoding
  #step_mutate_at(ACTION, fn = factor) |> 
  step_lencode_mixed(all_nominal_predictors(), 
                     outcome = vars(ACTION)) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_pca(all_predictors(), threshold = 0.8) |> 
  step_smote(all_outcomes(), neighbors = 10)

#target encoding (must
# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = trainData)


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
                            levels = 5)

## Set up K-fold CV
folds <- vfold_cv(trainData, v = 5, repeats = 1)

CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = NULL)

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best("accuracy")


## Predict
final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)


amazon_naive_bayes_predictions <- 
  predict(final_wf,
          new_data = testData,
          type = "prob") |> 
  bind_cols(testData) |> 
  #rename(ACTION = .pred_1) |> 
  select(id, type)

vroom_write(x = amazon_naive_bayes_predictions, 
            file="./AmazonNaiveBayesPreds.csv", delim=",")