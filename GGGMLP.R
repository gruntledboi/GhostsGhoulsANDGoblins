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

trainData$type <- as.factor(trainData$type)

nn_recipe <- recipe(formula = type ~ ., data = trainData) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>% ## Turn color to factor then dummy encode color
  step_dummy(color) |>
  step_range(all_numeric_predictors(), min = 0, max = 1) #scale to [0,1]

nn_model <- 
  mlp(hidden_units = tune(),
                epochs = 50) %>% #or 100 or 250
  set_engine("keras") %>% #verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

maxHiddenUnits = 20

nn_tuneGrid <- grid_regular(hidden_units(range = c(1, maxHiddenUnits)),
                            levels = 5)

## Set up K-fold CV
folds <- vfold_cv(trainData, v = 5, repeats = 1)


tuned_nn <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuneGrid,
            metrics = NULL)

tuned_nn %>% collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x = hidden_units, y = mean)) + geom_line()

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want