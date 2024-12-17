library(tidymodels)
library(keras)
library(remotes)
library(bonsai)
library(lightgbm)
library(vroom)
library(dbarts)

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

trainData$type <- as.factor(trainData$type)

my_recipe <- recipe(formula = type ~ ., data = trainData) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>% ## Turn color to factor then dummy encode color
  step_dummy(color) |>
  step_range(all_numeric_predictors(), min = 0, max = 1) #scale to [0,1]

boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

boost_tuneGrid <- grid_regular(tree_depth(), trees(), learn_rate(),
                              levels = 5)

## Set up K-fold CV
folds <- vfold_cv(trainData, v = 5, repeats = 1)


tuned_boost <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = boost_tuneGrid,
            metrics = metric_set(accuracy))

#######################

bestTune <- tuned_boost %>%
  select_best(metric = "accuracy")


## Predict
final_wf <-
  boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)


ggg_boost_predictions <- 
  predict(final_wf,
          new_data = testData,
          type = "class") |> 
  bind_cols(testData) |> 
  rename(type = .pred_class) |> 
  select(id, type)

vroom_write(x = ggg_bart_predictions, 
            file="./GGGBoostPreds.csv", delim=",")