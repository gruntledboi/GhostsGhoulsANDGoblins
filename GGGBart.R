# install.packages("remotes")
# remotes::install_github("rstudio/tensorflow")
# 
# reticulate::install_python()
# 
# install.packages("keras")
# keras::install_keras()
library(tidymodels)
library(keras)
library(remotes)
library(bonsai)
library(lightgbm)
library(vroom)
library(dbarts)

#install.packages("lightgbm")

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

# bart_model <- parsnip::bart(trees = tune()) %>% # BART figures out depth and learn_rate
#   set_engine("dbarts") %>% # might need to install
#   set_mode("classification")

#maxHiddenUnits = 20

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

bart_tuneGrid <- grid_regular(trees(),
                            levels = 5)

## Set up K-fold CV
folds <- vfold_cv(trainData, v = 5, repeats = 1)


tuned_bart <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = bart_tuneGrid,
            metrics = metric_set(accuracy))

# tuned_bart %>% collect_metrics() %>%
#   filter(.metric == "accuracy") %>%
#   ggplot(aes(x = hidden_units, y = mean)) + geom_line()

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want

#START WORK HERE
bestTune <- tuned_bart %>%
  select_best(metric = "accuracy")


## Predict
final_wf <-
  bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)


ggg_bart_predictions <- 
  predict(final_wf,
          new_data = testData,
          type = "class") |> 
  bind_cols(testData) |> 
  rename(type = .pred_class) |> 
  select(id, type)

vroom_write(x = ggg_bart_predictions, 
            file="./GGGBartPreds.csv", delim=",")