supervised_ml <- function(
    modeling_data = data,               # The data to be analyzed
    model_type = "lgbm",                # The ML model, can be "rf", "xgb", "lr", "lgbm", "svm"
    mode = "classification",            # The mode of analysis, can be "classification" or "regression"
    id_var = "Code",                    # Any id variable in data that is not part of analysis
    target = "PIM_Encounter",           # The target variable
    event_level = "Yes",                # The label of event in case of categorical target variable, can be "Yes" or "1" or anything else
    control_level = "No",               # The label of control in case of categorical target variable, can be "No" or "0" or anything else
    num_cols = 2:6,                     # Column indices of the numerical predictors
    cat_cols = 7:22,                    # Column indices of the categorical predictors
    train_split = 0.8,                  # Ratio of split of training and testing datasets
    cv_nfolds = 10,                     # Number of folds of Cross-validation
    cv_repeats = 10,                    # Number of repeats of cross-validation
    n_grid = 100,                       # Number of grids through which hyperparameter is to be tuned
    n_boot_metrics = 500,               # Number of bootstrap for estimation of metrics confidence intervals
    nsim_shap = 100,                    # Number of repeats during SHAP estimation
    top_n_features = NULL,              # Top N features to prune the dataset
    prefix = "LGBM",                    # Prefix for name of the output files
    output_dir = "LGBM"                 # Name of directory to save output files
) { 
  # 1. Load necessary libraries ---------------
  pacman::p_load(
    # --- 1. Core Framework & Data Manipulation ---
    tidymodels, # Meta-package for the modeling workflow
    dplyr, # Data manipulation
    stringr, # String handling
    glue, # Dynamic string formatting
    scales, # Scaling functions for plots (e.g., rescale)

    # --- 2. Modeling Engines (The "Workers") ---
    xgboost, # Gradient boosting
    lightgbm, # Fast gradient boosting
    bonsai, # Interface for LightGBM in tidymodels
    ranger, # Fast Random Forest
    glmnet, # Lasso/Ridge/Elastic Net (Logistic/Linear)
    kernlab, # Support Vector Machines

    # --- 3. Feature Engineering & Tuning ---
    finetune, # Racing methods for hyperparameter tuning
    themis, # Dealing with class imbalance (SMOTE/Downsampling)
    FSelectorRcpp, # Information gain / Feature selection
    lsr, # For etaSquared (Categorical feature selection in regression) 
    purrr, # For function map_dfr
    energy, # For dcor (Distance Correlation)

    # --- 4. Model Evaluation & Metrics ---
    yardstick, # Tidyverse metrics (AUC, RMSE, etc.)
    pROC, # ROC curve analysis
    PRROC, # Precision-Recall curves
    dcurves, # Decision Curve Analysis (DCA)
    lmtest, # Likelihood ratio tests / statistical diagnostics

    # --- 5. Interpretability (XAI) ---
    vip, # Variable Importance Plots
    fastshap, # Rapid SHAP value calculation
    shapviz, # Beautiful SHAP visualizations

    # --- 6. Visualization & Reporting ---
    ggplot2, # Core plotting
    patchwork, # Combining multiple plots (p1 + p2)
    openxlsx, # Exporting results to Excel
    stats,   # For saving the model

    # --- 7. Infrastructure ---
    doParallel, # Multicore processing
    caret # Included for legacy utilities (though tidymodels replaces it)
  )

  # 2. Setup Data ---------------------
  modeling_data <- modeling_data %>%
    mutate(across(all_of(cat_cols), as.factor)) %>%
    mutate(across(all_of(num_cols), as.numeric))

  if (mode == "classification") {
    modeling_data[[target]] <- factor(modeling_data[[target]],
      levels = c(event_level, control_level)
    )
  } else {
    modeling_data[[target]] <- as.numeric(modeling_data[[target]])
  }

  pred_col <- paste0(".pred_", event_level)
  cell_split <- initial_split(modeling_data,
    prop = train_split,
    strata = all_of(target)
  )
  cell_train <- training(cell_split)
  cell_test <- testing(cell_split)

  # 3. Data pre-processing  -----------
  ## 3.1 Feature engineering by filter method -------------
  if (mode == "classification") {
  ### 3.1.1 Classification ------
    feature_selection_results <- data.frame(
      feature = information_gain(as.formula(paste(target, "~ .")),
        data = cell_train %>% select(-all_of(id_var)),
        type = "infogain"
      )$attributes,
      a = information_gain(as.formula(paste(target, "~ .")),
        data = cell_train %>% select(-all_of(id_var)),
        type = "infogain"
      )$importance,
      b = information_gain(as.formula(paste(target, "~ .")),
        data = cell_train %>% select(-all_of(id_var)),
        type = "gainratio"
      )$importance,
      c = information_gain(as.formula(paste(target, "~ .")),
        data = cell_train %>% select(-all_of(id_var)),
        type = "symuncert"
      )$importance
    ) %>%
      mutate(overall_score = (a * b * c)^(1 / 3)) %>%
      arrange(desc(overall_score))
  } else {
    ### 3.1.2 Regression (Enhanced for Categorical & Numeric) ----------
    target_vec <- cell_train[[target]]
    features_to_test <- cell_train %>% select(-all_of(c(id_var, target)))
    feature_selection_results <- map_dfr(names(features_to_test), function(feat_name) {
      feat_vec <- features_to_test[[feat_name]]
      # Logic for Numeric Predictors (Correlation)
      if(is.numeric(feat_vec)) {
        temp_df <- data.frame(v1 = feat_vec, v2 = target_vec) %>% filter(complete.cases(.))
        a_val <- abs(cor(temp_df$v1, temp_df$v2, method = "pearson"))
        b_val <- abs(cor(temp_df$v1, temp_df$v2, method = "spearman"))
        c_val <- energy::dcor(temp_df$v1, temp_df$v2)
      } else {
        # Logic for Categorical Predictors (Eta-squared via ANOVA)
        # This measures how much variance in the target is explained by the category
        mod <- lm(target_vec ~ feat_vec)
        eta_sq <- lsr::etaSquared(mod)[1]
        a_val <- eta_sq
        b_val <- eta_sq
        c_val <- eta_sq
      }
      tibble(feature = feat_name, a = a_val, b = b_val, c = c_val)
    }) %>%
      mutate(overall_score = (a * b * c)^(1 / 3)) %>%
      arrange(desc(overall_score))
  }

  ### 3.1.3 Optional pruning logic ----------
  if (!is.null(top_n_features)) {

    # Identify which features to keep
    keep_features <- feature_selection_results$feature[1:min(top_n_features, nrow(feature_selection_results))]

    # Update the training and test sets
    cell_train <- cell_train %>% select(all_of(id_var), all_of(target), all_of(keep_features))
    cell_test  <- cell_test  %>% select(all_of(id_var), all_of(target), all_of(keep_features))
  }
  
  # 3.2 Recipe for pre-processing ---------------
  cell_rec <- recipe(as.formula(paste(target, "~ .")), data = cell_train) %>%
    update_role(all_of(id_var), new_role = "id") %>% # To specify id
    step_other(all_nominal_predictors(), threshold = 0.05) %>% # Collapse rare categories into other
    step_novel(all_nominal_predictors()) %>% # Handle categories in test data that weren't in training
    step_dummy(all_nominal_predictors()) %>% # create dummy variables
    step_zv(all_predictors()) %>% # Remove zero-variance variables
    step_normalize(all_numeric_predictors()) %>% # Normalize all the numeric predictors (not needed for RF)
    step_impute_knn(all_predictors(), neighbors = 5) # Imputing missing values based on KNN method
  # ONLY apply SMOTE if classification
  if (mode == "classification") {
    cell_rec <- cell_rec %>% step_smote(all_of(target))
  }

  # To check the final recipe
  prepped_rec <- prep(cell_rec)

  # 4. Create 10x10 Fold Cross-Validation (100 total re-samples) ----------------
  set.seed(123)
  folds_10x10 <- vfold_cv(cell_train,
                          v = cv_nfolds,
                          repeats = cv_repeats,
                          strata = all_of(target)
  )
  
  # 5. Model Specification with tune() placeholders ----------------
  ## 5.1 Random forest classification ---------------
  rf_spec <- rand_forest(
    mtry = tune(), # Number of predictors
    min_n = tune(), # Minimum data points in a node
    trees = tune()
  ) %>%
    set_engine("ranger",
      importance = "permutation",
      splitrule = tune(),
      sample.fraction = tune()
    ) %>%
    set_mode(mode)

  ## 5.2 Xgboost classification -----------------
  xgb_spec <- boost_tree(
    trees = tune(),
    tree_depth = tune(), # How deep the trees grow
    min_n = tune(), # Minimum data points in a node
    loss_reduction = tune(), # Early pruning (gamma)
    sample_size = tune(), # Sub-sampling (stochasticity)
    mtry = tune(), # Number of predictors
    learn_rate = tune() # Step size shrinkage
  ) %>%
    set_engine("xgboost") %>%
    set_mode(mode)

  ## 5.3 LightGBM classification -----------------
  lgbm_spec <- boost_tree(
    trees = tune(),
    tree_depth = tune(), # How deep the trees grow
    min_n = tune(), # Minimum data points in a node
    loss_reduction = tune(), # Early pruning (gamma)
    sample_size = tune(), # Sub-sampling (stochasticity)
    mtry = tune(), # Number of predictors
    learn_rate = tune() # Step size shrinkage
  ) %>%
    set_engine("lightgbm") %>%
    set_mode(mode)

  ## 5.4 Penalized Logistic Regression -----------------
  if (mode == "classification") {
    lr_spec <- logistic_reg(
      penalty = tune(),
      mixture = tune()
    ) %>%
      set_engine("glmnet") %>%
      set_mode(mode)
  } else {
    lr_spec <- linear_reg(
      penalty = tune(),
      mixture = tune()
    ) %>%
      set_engine("glmnet") %>%
      set_mode(mode)
  }

  ## 5.5 Support Vector Machine (Radial Basis Function) -----------------
  svm_spec <- svm_rbf(
    cost = tune(),
    margin = tune(),
    rbf_sigma = tune()
  ) %>%
    set_engine("kernlab") %>%
    set_mode(mode)

  # 6. Workflow bundle ------------------
  rf_wflow <- workflow() %>%
    add_model(rf_spec) %>%
    add_recipe(cell_rec)
  xgb_wflow <- workflow() %>%
    add_model(xgb_spec) %>%
    add_recipe(cell_rec)
  lgbm_wflow <- workflow() %>%
    add_model(lgbm_spec) %>%
    add_recipe(cell_rec)
  lr_wflow <- workflow() %>%
    add_model(lr_spec) %>%
    add_recipe(cell_rec)
  svm_wflow <- workflow() %>%
    add_model(svm_spec) %>%
    add_recipe(cell_rec)

  # 7. Tuning -------
  ## 7.1 Parameter sets --------------
  rf_param_set <- extract_parameter_set_dials(rf_wflow) %>%
    update(sample.fraction = sample_prop(range = c(0.1, 0.9))) %>%
    finalize(bake(prepped_rec, new_data = NULL, has_role("predictor"))) # Finalizes mtry based on dummy vars

  xgb_params <- extract_parameter_set_dials(xgb_wflow) %>%
    finalize(bake(prepped_rec, new_data = NULL, has_role("predictor")))

  lgbm_params <- extract_parameter_set_dials(lgbm_wflow) %>%
    finalize(bake(prepped_rec, new_data = NULL, has_role("predictor")))

  lr_params <- extract_parameter_set_dials(lr_wflow)

  svm_params <- extract_parameter_set_dials(svm_wflow) %>%
    finalize(bake(prepped_rec, new_data = NULL, has_role("predictor")))

  ## Toggle between models ---------# Options: "rf", "xgb", "lgbm", "lr", "svm"
  active_wflow <- if (model_type == "rf") rf_wflow else if (model_type == "xgb") xgb_wflow else if (model_type == "lgbm") lgbm_wflow else if (model_type == "lr") lr_wflow else svm_wflow

  active_params <- if (model_type == "rf") rf_param_set else if (model_type == "xgb") xgb_params else if (model_type == "lgbm") lgbm_params else if (model_type == "lr") lr_params else svm_params

  ## 7.2 Run the Race ------------
  registerDoParallel(cores = parallel::detectCores(logical = FALSE))
  set.seed(456)
  eval_metrics <- if (mode == "classification") {
    metric_set(pr_auc, f_meas)
  } else {
    metric_set(rmse, rsq, mae)
  }
  race_results <- tune_race_anova(
    object = active_wflow,
    resamples = folds_10x10,
    grid = n_grid,
    param_info = active_params,
    metrics = eval_metrics,
    control = control_race(verbose = TRUE, save_pred = TRUE, burn_in = 3)
  )
  stopImplicitCluster()

  ## 7.3 Visualize the racing process --------
  target_metric <- if (mode == "classification") {
    "pr_auc"
  } else {
    "rmse"
  }
  show_best(race_results,
    metric = target_metric
  )
  best_config <- select_best(race_results, metric = target_metric)
  cv_fold_data <- collect_metrics(race_results, summarize = FALSE) %>%
    filter(.config == best_config$.config)
  
  # 8. Evaluate results ----------
  ## 8.1 Select the winner and finalize -------
  best_params_raw <- select_best(race_results,
    metric = target_metric
  )

  hp_definitions <- tribble(
    ~parameter, ~definition, ~technical_meaning, ~clinical_impact,
    "mtry", "Number of predictors sampled at each split.", "Determines the degree of randomness in tree construction.", "High mtry allows dominant clinical features to lead every tree; low mtry forces the model to find 'hidden' patterns in less obvious variables.",
    "trees", "Total number of trees in the ensemble.", "The size of the committee of trees.", "More trees lead to more stable risk predictions across different patient sub-groups, but significantly increase processing time.",
    "min_n", "Minimum data points in a node.", "Controls the granularity of the 'leaves' (endpoints) of the trees.", "Large values prevent the model from 'memorizing' rare individual cases (overfitting), creating a more generalized model for the whole population.",
    "tree_depth", "Maximum depth of each individual tree.", "Limits the length of the decision path from root to leaf.", "Deep trees (8+) capture complex patient 'profiles' (e.g., Age + Comorbidity + Lab value interaction); shallow trees (1-3) only look at simple main effects.",
    "learn_rate", "Step size shrinkage (Eta).", "Determines the contribution of each new tree to the final prediction.", "A slower rate (0.01) is safer and usually more accurate but requires more 'trees' to learn the relationship between features and the outcome.",
    "loss_reduction", "Minimum loss reduction (Gamma).", "A threshold for whether a split is 'worth it' based on error improvement.", "A high value acts as a conservative filter, only allowing the model to split on variables that provide a clear, statistically meaningful improvement.",
    "sample_size", "Proportion of data used for each tree (Boosting).", "Percentage of training patients seen by each individual tree.", "Lower values (e.g., 0.70) introduce 'stochasticity,' helping the model remain robust even if the training data has some noisy or outlier patient records.",
    "sample.fraction", "Proportion of data used for each tree (RF).", "Same concept as sample_size, specifically for the 'ranger' engine.", "Helps decorrelate trees; if every tree sees all patients, they might all make the same errors.",
    "splitrule", "Criterion used to select variable splits.", "The mathematical logic (Gini, Extratrees, etc.) used to evaluate a split.", "Determines if the model prioritizes 'purity' of the resulting groups or a more randomized approach to speed up training.",
    "penalty", "Total regularization (Lambda).", "The 'strength' of the penalty applied to large coefficients.", "High penalty results in a simpler 'clinical rule' by shrinking less important variables toward zero; low penalty allows for a more complex (but potentially fragile) model.",
    "mixture", "Proportion of Lasso (1) vs Ridge (0).", "Determines the 'flavor' of the penalty.", "A value of 1.0 (Lasso) performs automatic feature selection—it will literally delete irrelevant variables from the model equation.",
    "cost", "Cost of constraint violation.", "The 'budget' allowed for misclassified points near the boundary.", "High cost creates a 'hard' boundary that tries to get every patient right; low cost allows a 'soft' boundary for better generalization to new data.",
    "margin", "Insensitivity zone.", "The 'buffer' area where errors are not penalized.", "Larger margins ignore 'small errors' in patient probability, focusing only on clear, high-confidence classifications.",
    "rbf_sigma", "Reach of the kernel (Gaussian).", "Determines how 'local' the influence of a single patient's data point is.", "High sigma means a patient only influences the risk score of very similar patients (high complexity); low sigma means their influence spreads further (smoother boundary)."
  )

  best_params <- best_params_raw %>%
    mutate(across(everything(), as.character)) %>%
    pivot_longer(
      cols = everything(),
      names_to = "parameter",
      values_to = "value"
    ) %>%
    left_join(hp_definitions, by = "parameter") %>%
    mutate(model_type = !!model_type) %>%
    select(model_type, parameter, value, definition, technical_meaning, clinical_impact)

  final_wf <- finalize_workflow(active_wflow, best_params_raw)
  eval_metrics_test <- if (mode == "classification") {
    metric_set(
      roc_auc,
      pr_auc,
      brier_class,
      kap,
      accuracy,
      f_meas,
      mn_log_loss,
      yardstick::recall,
      yardstick::sensitivity,
      yardstick::specificity,
      yardstick::precision
    )
  } else {
    metric_set(
      rmse,
      rsq,
      mae,
      mape,
      huber_loss
    )
  }

  final_fit <- last_fit(final_wf, cell_split, metrics = eval_metrics_test)

  ## 8.2 Extract predictions ----------------
  final_preds <- collect_predictions(final_fit)
  if (mode == "classification") {
    actuals_numeric <- as.numeric(final_preds[[target]] == event_level)
    probs <- final_preds[[pred_col]]
  } else {
    actuals_numeric <- final_preds[[target]]
    probs <- final_preds[[".pred"]]
  }

  if (mode == "classification") {
  
  ## 8.3 Threshold optimization (Youden's J Statistic) ---------
    roc_obj_j <- pROC::roc(actuals_numeric, probs, quiet = TRUE)
    best_threshold_data <- pROC::coords(roc_obj_j, x = "best", best.method = "youden")
    optimal_threshold <- best_threshold_data$threshold
    max_j_index <- best_threshold_data$sensitivity + best_threshold_data$specificity - 1
    sensitivity_threshold <- best_threshold_data$sensitivity
    specitivity_threshold <- best_threshold_data$specificity

    threshold_summary <- data.frame(
      Metric = c(
        "Optimal Threshold",
        "Max J-Index",
        "Sensitivity at Threshold",
        "Specitivity at Threshold"
      ),
      Value = c(
        optimal_threshold,
        max_j_index,
        sensitivity_threshold,
        specitivity_threshold
      )
    )

    ## 8.4 Create the "Adjusted" classes based on J-Threshold ---------------------
    final_preds <- final_preds %>%
      mutate(.pred_class_j = factor(
        ifelse(.[[pred_col]] >= optimal_threshold, event_level, control_level),
        levels = c(event_level, control_level)
      ))
  }

 ## 8.5 Metrics and bootstrap CIS ---------
  set.seed(123)
  boot_metrics <- bootstraps(cell_test, times = n_boot_metrics) %>%
    mutate(metrics = map(splits, ~ {
      preds <- augment(extract_workflow(final_fit), as.data.frame(.x))
      if (mode == "classification") {
        preds <- preds %>%
          mutate(.pred_class_j = factor(
            ifelse(.[[pred_col]] >= optimal_threshold, event_level, control_level),
            levels = c(event_level, control_level)
          ))
        res <- eval_metrics_test(preds, truth = all_of(target), estimate = .pred_class_j, !!sym(pred_col))
        f2_val <- f_meas(preds, truth = all_of(target), estimate = .pred_class_j, beta = 2) %>%
          mutate(.metric = "f2_score")
        return(bind_rows(res, f2_val))
      } else {
        return(eval_metrics_test(preds, truth = all_of(target), estimate = .pred))
      }
    })) %>%
    unnest(metrics)

  ci_values <- boot_metrics %>%
    group_by(.metric) %>%
    summarise(
      mean_est = mean(.estimate, na.rm = TRUE),
      lower_ci = quantile(.estimate, 0.025, na.rm = TRUE),
      upper_ci = quantile(.estimate, 0.975, na.rm = TRUE)
    )

  if (mode == "classification") {
    ci_values <- ci_values %>%
      bind_rows(tibble(
        .metric = "prevalance",
        mean_est = mean(final_preds[[target]] == event_level),
        lower_ci = NA,
        upper_ci = NA
      ))
  }

  metric_interpretations <- if (mode == "classification") {
    tribble(
      ~.metric, ~Metric_name, ~Interpretation, ~Clinical_Meaning,
      "roc_auc", "ROC-AUC", "0.5=Random; 0.7-0.8=Acceptable; >0.8=Excellent.", "Ability to rank risk: Probability that a random positive patient has a higher risk score than a random negative one.",
      "pr_auc", "PR-AUC", "Must be > Class Prevalence to be useful.", "Average precision; critical for identifying cases when the event is rare.",
      "accuracy", "Accuracy", "Compare against No-Information Rate (NIR).", "The percentage of all predictions (Yes and No) that were correct.",
      "sensitivity", "Sensitivity (Recall)", "High (>0.8) is better for screening.", "The ability to correctly identify patients who actually had the event.",
      "specificity", "Specificity", "High (>0.8) is better to avoid false alarms.", "The ability to correctly identify patients who did NOT have the event.",
      "precision", "Precision (PPV)", "High (>0.7) means fewer false positives.", "If the model flags a patient, how likely is it they actually had the event?",
      "f_meas", "F1-Score", "0.0 is poor; >0.7 is strong.", "Harmonic mean of Precision/Sensitivity; assumes equal cost for false positives and misses.",
      "f2_score", "F2-Score", "Values > prevalence; >0.6 is good.", "Weights Sensitivity twice as high as Precision (prioritizing finding cases).",
      "kap", "Cohen’s Kappa", "0.0=Chance; 0.6-0.8=Substantial.", "Agreement beyond chance; accounts for guessing correctly by luck.",
      "brier_class", "Brier Score", "0.0 is perfect; <0.25 is required.", "Mean squared difference between predicted probability and actual outcome.",
      "mn_log_loss", "Log Loss", "Lower values = more confident, correct guesses.", "Punishes the model for being confident but wrong.",
      "recall", "Recall", "High (>0.8) is better for screening.", "Fraction of actual events successfully 'called' by the model.",
      "prevalance", "Baseline Prevalence", "PR-AUC and F2 Score must be higher", "The actual rate of events in data"
    )
  } else {
    tribble(
      ~.metric, ~Metric_name, ~Interpretation, ~Clinical_Meaning,
      "rmse", "RMSE", "Lower is better (Goal: < Clinical SD)", "The typical 'margin of error' to expect for a single patient.",
      "rsq", "R-Squared", "Higher is better (Goal: >0.70)", "The % of patient outcome variation captured by the model.",
      "mae", "MAE", "Lower is better (Goal: < RMSE)", "The average 'miss' distance per patient, ignoring extremes.",
      "mape", "MAPE", "Lower is better (Goal: <10%)", "The average % error relative to the actual patient value.",
      "huber_loss", "Huber Loss", "Lower is better", "Model reliability when handling complex, outlier-heavy data."
    )
  }

  ci_values <- ci_values %>%
    left_join(metric_interpretations, by = ".metric") %>%
    mutate(
      Performance_Tier = case_when(
        # Logic for AUC-based metrics (Higher is better)
        mode == "classification" & .metric %in% c("roc_auc", "pr_auc") & mean_est >= 0.9 ~ "Outstanding",
        mode == "classification" & .metric %in% c("roc_auc", "pr_auc") & mean_est >= 0.8 ~ "Excellent",
        mode == "classification" & .metric %in% c("roc_auc", "pr_auc") & mean_est >= 0.7 ~ "Acceptable",
        mode == "classification" & .metric %in% c("roc_auc", "pr_auc") ~ "Poor",

        # Logic for Error-based metrics (Lower is better)
        mode == "classification" & .metric == "brier_class" & mean_est <= 0.1 ~ "Highly Accurate",
        mode == "classification" & .metric == "brier_class" & mean_est <= 0.25 ~ "Acceptable",
        mode == "classification" & .metric == "brier_class" ~ "Inaccurate",

        # Logic for Agreement/Classification metrics
        mode == "classification" & .metric %in% c("accuracy", "sensitivity", "specificity", "precision", "f_meas", "f2_score") & mean_est >= 0.8 ~ "Strong",
        mode == "classification" & .metric %in% c("accuracy", "sensitivity", "specificity", "precision", "f_meas", "f2_score") & mean_est >= 0.6 ~ "Moderate",

        # Logic for Kappa
        mode == "classification" & .metric == "kap" & mean_est >= 0.6 ~ "Substantial Agreement",
        mode == "classification" & .metric == "kap" & mean_est >= 0.4 ~ "Moderate Agreement",

        # --- Regression Tiers (Driven primarily by R-Squared) ---
        mode == "regression" & .metric == "rsq" & mean_est >= 0.85 ~ "Elite",
        mode == "regression" & .metric == "rsq" & mean_est >= 0.70 ~ "Excellent",
        mode == "regression" & .metric == "rsq" & mean_est >= 0.50 ~ "Good",
        mode == "regression" & .metric == "rsq" & mean_est < 0.50 ~ "Fair/Poor",

        # --- MAPE Tiers (Percentage Error) ---
        mode == "regression" & .metric == "mape" & mean_est <= 0.05 ~ "Elite (<5% error)",
        mode == "regression" & .metric == "mape" & mean_est <= 0.15 ~ "Good (<15% error)",
        TRUE ~ "Supporting Metric"
      )
    ) %>%
    mutate(across(c(mean_est, lower_ci, upper_ci), ~ {
      if(is.numeric(.)) round(., 3) else .
    })) %>% 
    select(.metric, mean_est, lower_ci, upper_ci, Performance_Tier, Metric_name, Interpretation, Clinical_Meaning) %>%
    relocate(.metric, Metric_name, mean_est, lower_ci, upper_ci, Performance_Tier, Interpretation, Clinical_Meaning)

  # 9. Tests ---------
  if (mode == "classification") {
    ## 9.1 Classification: McNemar and Accuracy P-Value --------
    cm_stats <- caret::confusionMatrix(
      data = final_preds$.pred_class_j,
      reference = final_preds[[target]],
      positive = event_level
    )
    test_p1 <- cm_stats$overall["McnemarPValue"]
    test_p2 <- cm_stats$overall["AccuracyPValue"]
    test_name1 <- "McNemar's Test P-value (Bias Check)"
    test_name2 <- "Accuracy vs. NIR P-value"

    ## 9.2 Spiegelhalter Z-test (Calibration) --- -------
    spiegelhalter_p_calc <- function(final_preds, target) {
      y_true <- as.numeric(final_preds[[target]] == event_level)
      y_prob <- final_preds[[pred_col]]
      y_prob <- pmax(pmin(y_prob, 1 - 1e-9), 1e-9)
      res <- y_true - y_prob
      z_stat <- sum((1 - 2 * y_prob) * res, na.rm = TRUE) /
        sqrt(sum((1 - 2 * y_prob)^2 * y_prob * (1 - y_prob), na.rm = TRUE))
      p_value <- 2 * pnorm(-abs(z_stat))
      return(p_value)
    }
    test_p3 <- spiegelhalter_p_calc(final_preds, target)
    test_name3 <- "Spiegelhalter Calibration test P-value"
  } else {
    ## 9.1 Regression Specific: Breusch-Pagan & Ramsey RESET --------
    formula_str <- reformulate(".pred", response = paste0("as.numeric(", target, ")"))
    reg_diag_lm <- lm(formula_str, data = final_preds)
    test_p1 <- lmtest::bptest(reg_diag_lm)$p.value
    test_p2 <- lmtest::resettest(reg_diag_lm, power = 2:3, type = "fitted")$p.value
    test_name1 <- "Breusch-Pagan Test(Heteroscedasticity) P-value"
    test_name2 <- "Ramsey RESET Test (functional form) P-value"

    # 9.2 Regression Calibration (Durbin-Watson for Autocorrelation) ------------------
    test_p3 <- lmtest::dwtest(reg_diag_lm)$p.value
    test_name3 <- "Durbin-Watson (AutoCorr) P-value"
  }

  ## 9.3 Permutation Test (Real Signal vs Noise) --- ----------
  set.seed(123)
  perm_vals <- replicate(500, {
    shuffled_labels <- sample(final_preds[[target]])
    if (mode == "classification") {
      as.numeric(yardstick::roc_auc_vec(shuffled_labels, final_preds[[pred_col]]))
    } else {
      cor(shuffled_labels, final_preds$.pred)
    }
  })

  real_vals <- if (mode == "classification") {
    collect_metrics(final_fit) %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
  } else {
    cor(final_preds[[target]], final_preds$.pred)
  }

  permutation_p <- mean(abs(perm_vals) >= abs(real_vals))

  ## 9.4 Cook's Distance (Influential Cases) ---------
  diag_formula <- if (mode == "classification") {
    as.formula(paste(target, "~", pred_col))
  } else {
    as.formula(paste(target, "~ .pred"))
  }

  diag_model <- if (mode == "classification") {
    glm(diag_formula, data = final_preds, family = binomial)
  } else {
    lm(diag_formula, data = final_preds)
  }

  cooks_dist <- cooks.distance(diag_model)
  influential_obs_count <- sum(cooks_dist > (4 / nrow(final_preds)), na.rm = TRUE)

  ## 9.5 Diagnostic summary --------
  diagnostic_summary <- data.frame(
    Metric = c(
      test_name1,
      test_name2,
      test_name3,
      "AUC Permutation Test P-value (Signal Check)",
      "Influential Cases (Cook's Distance)"
    ),
    Value = c(
      test_p1,
      test_p2,
      test_p3,
      permutation_p,
      as.numeric(influential_obs_count)
    )
  ) %>%
    mutate(Interpretation = case_when(
      Metric == "Breusch-Pagan Test(Heteroscedasticity) P-value" ~ if_else(Value > 0.05, "Pass: Constant error variance", "Fail: Errors vary with predicted value"),
      Metric == "Ramsey RESET Test (functional form) P-value" ~ if_else(Value > 0.05, "Pass: Linear form sufficient", "Fail: Consider non-linear model"),
      Metric == "Durbin-Watson (AutoCorr) P-value" ~ if_else(Value > 1.5 & Value < 2.5, "Pass: No autocorrelation", "Fail: Residuals are correlated"),
      Metric == "McNemar's Test P-value (Bias Check)" ~ if_else(Value > 0.05, "Pass: No systematic bias", "Fail: Systematic error found"),
      Metric == "Accuracy vs. NIR P-value" ~ if_else(Value < 0.05, "Pass: Better than base rate", "Fail: Not better than guessing"),
      Metric == "Spiegelhalter Calibration test P-value" ~ if_else(Value > 0.05, "Pass: Probabilities reliable", "Fail: Unreliable risk scores"),
      Metric == "AUC Permutation Test P-value (Signal Check)" ~ if_else(Value < 0.05, "Pass: Real signal detected", "Fail: Likely noise/overfit"),
      Metric == "Influential Cases (Cook's Distance)" ~ paste(Value, "cases > 4/n (Check outliers)"),
      TRUE ~ "Review required"
    ))

  # 10. Variable Importance ----------
  clean_label <- function(x) {
    x %>%
      gsub("_", " ", .) %>%
      stringr::str_trim() %>%
      stringr::str_squish()
  }

  # Get the names of the categorical columns
  cat_names <- colnames(cell_train)[cat_cols]
  cat_pattern <- paste0("^(", paste(cat_names, collapse = "|"), ")_")

  # Extract the engine once
  final_engine <- extract_fit_parsnip(final_fit)

  final_fitted_wf <- extract_workflow(final_fit)
  train_processed <- bake(prep(cell_rec), has_role("predictor"), new_data = cell_train)
  test_processed <- bake(prep(cell_rec), has_role("predictor"), new_data = cell_test)

  # Define a prediction wrapper function
  p_fun <- function(object, newdata) {
    if (mode == "classification") {
      predict(object, as.data.frame(newdata), type = "prob")[[pred_col]]
    } else {
      predict(object, as.data.frame(newdata))[[".pred"]]
    }
  }

  ## 10.1 Importance ---------
  # (Model-Specific (Permutation for RF, SVM; Gain for XGB, LGBM; Beta coefficient for LR)
  vi_data <- tryCatch(
    {
      if (mode == "classification") {
        if (model_type == "lr") {
          vi(final_engine, lambda = best_params_raw$penalty)
        } else {
          vi(final_engine)
        }
      } else {
        # Regression VI
        vi(final_engine) # Most engines handle regression VI automatically
      }
    },
    error = function(e) {
      message("Model-specific VI failed. Switching to Permutation Importance...")
      vi_permute(
        final_engine,
        feature_names = colnames(train_processed),
        train = train_processed,
        target = cell_train[[target]],
        # DYNAMIC METRIC SELECTION
        metric = if (mode == "classification") "roc_auc" else "rmse",
        pred_wrapper = p_fun,
        nsim = 10
      )
    }
  )

  vi_data_dummy <- vi_data %>%
    mutate(Variable = clean_label(Variable)) %>%
    arrange(desc(abs(Importance)))

  vi_data_grouped <- vi_data %>%
    mutate(Variable = ifelse(
      str_detect(Variable, cat_pattern),
      str_extract(Variable, cat_pattern) %>% str_remove("_$"), # Extract parent, remove trailing _
      Variable # Keep numeric variables exactly as they are
    )) %>%
    group_by(Variable) %>%
    summarise(Importance = sum(abs(Importance)), .groups = "drop") %>%
    arrange(desc(Importance)) %>%
    mutate(Variable = clean_label(Variable))

  ## 10.2 SHAP based importance ----------
  # Compute SHAP values
  set.seed(789)
  shap_results <- explain(
    extract_fit_parsnip(final_fitted_wf),
    X = as.matrix(train_processed), # Background distribution
    newdata = as.matrix(test_processed), # Data to explain
    pred_wrapper = p_fun,
    nsim = nsim_shap, # Number of Monte Carlo simulations (increase for more precision)
    adjust = TRUE # Ensures SHAP values sum to the difference in predictions
  )

  shap_long <- shap_results %>%
    as.data.frame() %>%
    mutate(row_id = row_number()) %>%
    pivot_longer(-row_id, names_to = "feature", values_to = "shap_value") %>%
    mutate(Variable = clean_label(feature))

  # SHAP table with dummy variables
  shap_summary_table <- shap_long %>%
    group_by(Variable) %>% # Group by the pretty names
    summarise(
      n_obs = n(),
      mean_abs_shap = mean(abs(shap_value)), # Primary Ranking Metric
      mean_shap = mean(shap_value),
      sd_shap = sd(shap_value),
      min_shap = min(shap_value),
      max_shap = max(shap_value),
      prop_positive = mean(shap_value > 0),
      prop_negative = mean(shap_value < 0),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_abs_shap)) %>%
    mutate(rank = row_number()) %>%
    relocate(rank, Variable)

  # SHAP table with original variables
  shap_summary_table_grouped <- shap_long %>%
    mutate(Variable = ifelse(
      str_detect(feature, cat_pattern),
      str_extract(feature, cat_pattern) %>% str_remove("_$"), # Extract parent, remove trailing _
      feature # Keep numeric variables exactly as they are
    )) %>%
    mutate(Variable = clean_label(Variable)) %>%
    group_by(row_id, Variable) %>%
    summarise(shap_value = sum(shap_value), .groups = "drop") %>%
    group_by(Variable) %>%
    summarise(
      n_obs = n(),
      mean_abs_shap = mean(abs(shap_value)), # Primary Ranking Metric
      mean_shap = mean(shap_value),
      sd_shap = sd(shap_value),
      min_shap = min(shap_value),
      max_shap = max(shap_value),
      prop_positive = mean(shap_value > 0),
      prop_negative = mean(shap_value < 0),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_abs_shap)) %>%
    mutate(rank = row_number()) %>%
    relocate(rank, Variable)

  ## 10.3 Combined importance table --------
  imp_label <- case_when(
    model_type %in% c("rf", "svm") ~ "Importance (Permutation)",
    model_type %in% c("xgb", "lgbm") ~ "Importance (Gain)",
    # Update for Regression vs Classification
    model_type == "lr" & mode == "classification" ~ "Importance (Beta coefficient)",
    model_type == "lr" & mode == "regression" ~ "Importance (Coefficient)",
    TRUE ~ "Importance"
  )

  master_importance_table_dummy <- vi_data_dummy %>%
    left_join(shap_summary_table, by = "Variable") %>%
    arrange(rank) %>%
    rename(!!imp_label := Importance)

  master_importance_table_original <- vi_data_grouped %>%
    left_join(shap_summary_table_grouped, by = "Variable") %>%
    arrange(rank) %>%
    rename(!!imp_label := Importance)

  # 11. Plots -------------
  ## 11.1 Diagnostic plots --------------
  if (mode == "classification") {
    ### 11.1.1 Classification specific ----------
    ### ROC Plot --------
    p_roc <- ggroc(roc_obj_j, color = "#2c7bb6", linewidth = 1.2) +
      geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
      annotate("point",
        x = best_threshold_data$specificity,
        y = best_threshold_data$sensitivity,
        color = "red", size = 3
      ) +
      annotate("label",
        x = 0.4, y = 0.2,
        label = glue("AUC: {round(as.numeric(auc(roc_obj_j)), 3)}")
      ) +
      annotate("text",
        x = best_threshold_data$specificity + 0.1, y = best_threshold_data$sensitivity - 0.05,
        label = glue("Best Threshold: {round(optimal_threshold, 2)}"), color = "red"
      ) +
      theme_minimal() +
      labs(title = "A. ROC Curve (with Optimal Cutoff")

    ### PR Plot -------
    pr_curve_data <- PRROC::pr.curve(
      scores.class0 = probs[actuals_numeric == 1],
      scores.class1 = probs[actuals_numeric == 0], curve = TRUE
    )
    p_pr <- ggplot(as.data.frame(pr_curve_data$curve), aes(x = V1, y = V2)) +
      geom_line(color = "#d7191c", linewidth = 1.2) +
      labs(title = "B. PR Curve", x = "Recall", y = "Precision") +
      theme_minimal()

    ### Calibration Plot ------
    p_cal <- ggplot(data.frame(obs = actuals_numeric, pred = probs), aes(x = pred, y = obs)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      geom_smooth(method = "glm", method.args = list(family = "binomial"), color = "#2c7bb6") +
      stat_summary_bin(fun = "mean", bins = 10, geom = "point", color = "red") +
      theme_minimal() +
      labs(title = "C. Calibration Plot")

    ### Confusion Matrix Heatmap ------
    cm_table <- conf_mat(final_preds, truth = all_of(target), estimate = .pred_class_j)$table
    cm_df <- as.data.frame(cm_table)

    p_cm <- ggplot(cm_df, aes(x = Prediction, y = Truth, fill = Freq)) +
      geom_tile(color = "white", lwd = 1) +
      geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
      scale_fill_gradient(low = "#e5f5f9", high = "#2ca25f") +
      labs(
        title = "D. Adjusted Confusion Matrix",
        subtitle = glue("Threshold: {round(optimal_threshold, 2)}"),
        x = "Predicted Class", y = "Actual Class"
      ) +
      theme_minimal() +
      theme(panel.grid = element_blank(), legend.position = "none")

    ### Combine into Dashboard -------
    final_dashboard <- (p_roc + p_pr) / (p_cal + p_cm) +
      plot_annotation(title = "Final Model Diagnostics")
  } else {
    ### 11.1.2 Regression specific ------
    resids_df <- final_preds %>%
      mutate(.resid = !!sym(target) - .pred)
    #### Calibration Plot (Observed vs. Predicted) --------
    # Uses LOESS to show if the model drifts from perfect calibration
    p_fit <- ggplot(resids_df, aes(x = .pred, y = !!sym(target))) +
      geom_point(alpha = 0.3, color = "#2c7bb6") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
      geom_smooth(method = "loess", color = "#d7191c", fill = "#d7191c", alpha = 0.2) +
      theme_minimal() +
      labs(
        title = "A. Regression Calibration",
        subtitle = "Red line (Actual) should overlap dashed line (Perfect)",
        x = "Predicted Value", y = glue("Observed {target}")
      )

    #### Residuals vs. Predicted (Heteroscedasticity & Bias) ------------
    # Checks if error variance is constant (Visual Breusch-Pagan)
    p_res_fit <- ggplot(resids_df, aes(x = .pred, y = .resid)) +
      geom_point(alpha = 0.3, color = "#d7191c") +
      geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1) +
      geom_smooth(method = "loess", color = "black", se = TRUE, linetype = "dotted") +
      theme_minimal() +
      labs(
        title = "B. Residuals vs. Predicted",
        subtitle = "Check for non-random error patterns or 'funnel' shapes",
        x = "Predicted Value", y = "Residual (Error)"
      )

    #### Distribution of Residuals (Normality) --------------
    # Visual check for Shapiro-Wilk/KS tests
    p_res_hist <- ggplot(resids_df, aes(x = .resid)) +
      geom_histogram(bins = 30, fill = "#2c7bb6", color = "white", alpha = 0.8) +
      theme_minimal() +
      labs(
        title = "C. Error Distribution",
        subtitle = "Errors should be normally distributed around zero",
        x = "Residual (Error)", y = "Patient Count"
      )

    #### Normal Q-Q Plot (Outlier Impact) ------------
    # Shows if extreme errors are more frequent than expected
    p_qq <- ggplot(resids_df, aes(sample = .resid)) +
      stat_qq(alpha = 0.4, color = "#2c7bb6") +
      stat_qq_line(color = "red") +
      theme_minimal() +
      labs(
        title = "D. Normal Q-Q Plot",
        subtitle = "Points off the line indicate extreme clinical misses",
        x = "Theoretical Quantiles", y = "Sample Quantiles"
      )

    #### Combine into Dashboard ----------
    final_dashboard <- (p_fit + p_res_fit) / (p_res_hist + p_qq) +
      plot_annotation(
        title = "Final Model Diagnostics: Regression",
        subtitle = glue("Target Variable: {target} | Error analysis and calibration check"),
        theme = theme(plot.title = element_text(size = 16, face = "bold"))
      )
  }

  ## 11.2 Variable importance plots --------
  ### Gini/Gain based importance plots -----------
  # With dummy variables
  p_vip_dummy <- vi_data_dummy %>%
    slice_max(Importance, n = 15) %>% # Take top 15
    ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_col(fill = "#2c7bb6") +
    coord_flip() + # Makes it a horizontal bar chart
    theme_minimal() +
    labs(
      title = "Variable Importance (Gini/Gain)",
      x = NULL,
      y = "Importance Score"
    )
  # With original variables
  p_vip_grouped <- vi_data_grouped %>%
    slice_max(Importance, n = 15) %>% # Take top 15
    ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_col(fill = "#2c7bb6") +
    coord_flip() + # Makes it a horizontal bar chart
    theme_minimal() +
    labs(
      title = "Variable Importance (Gini/Gain)",
      x = NULL,
      y = "Importance Score"
    )

  ### SHAP based importance plots --------
  # with dummy variables
  shap_matrix_clean <- as.matrix(shap_results)
  colnames(shap_matrix_clean) <- clean_label(colnames(shap_matrix_clean))
  test_processed_clean <- as.data.frame(test_processed)
  colnames(test_processed_clean) <- clean_label(colnames(test_processed_clean))
  shp <- shapviz(shap_matrix_clean, X = test_processed_clean)

  p_shap <- sv_importance(shp,
    kind = "beeswarm",
    max_display = 15, # Show only top 15 to avoid a "squashed" plot
    alpha = 0.5, # Transparency helps if you have many patients
    size = 1.2
  ) + # Adjust point size
    scale_y_discrete(labels = clean_label) + # Apply your cleaning function
    scale_color_gradientn(colors = c("#4575b4", "gray90", "#d73027"), name = "Feature\nValue") +
    theme_minimal() +
    labs(
      title = "SHAP Beeswarm",
      x = "Impact on Probability (SHAP Value)",
      y = NULL
    )

  # With original variables
  shap_grouped_df <- shap_long %>%
    mutate(Original_Variable = ifelse(
      str_detect(feature, cat_pattern),
      str_extract(feature, cat_pattern) %>% str_remove("_$"), # Extract parent, remove trailing _
      feature # Keep numeric variables exactly as they are
    )) %>%
    mutate(Original_Variable = clean_label(Original_Variable)) %>%
    group_by(row_id, Original_Variable) %>%
    summarise(shap_value = sum(shap_value), .groups = "drop") %>%
    pivot_wider(names_from = Original_Variable, values_from = shap_value) %>%
    select(-row_id)

  shap_grouped_matrix <- as.matrix(shap_grouped_df)

  test_grouped_X <- test_processed %>%
    mutate(row_id = row_number()) %>%
    pivot_longer(-row_id, names_to = "feature", values_to = "val") %>%
    mutate(Original_Variable = ifelse(
      str_detect(feature, cat_pattern),
      str_extract(feature, cat_pattern) %>% str_remove("_$"), # Extract parent, remove trailing _
      feature # Keep numeric variables exactly as they are
    )) %>%
    mutate(Original_Variable = clean_label(Original_Variable)) %>%
    group_by(row_id, Original_Variable) %>%
    summarise(val = max(val), .groups = "drop") %>%
    pivot_wider(names_from = Original_Variable, values_from = val) %>%
    select(-row_id)

  test_grouped_X <- test_grouped_X[, colnames(shap_grouped_matrix)]

  shp_grouped <- shapviz(shap_grouped_matrix, X = as.data.frame(test_grouped_X))

  p_shap_grouped <- sv_importance(shp_grouped,
    kind = "beeswarm",
    max_display = 15
  ) +
    scale_y_discrete(labels = clean_label) +
    scale_color_gradientn(
      colors = c("#4575b4", "gray90", "#d73027"),
      name = "Feature\nValue"
    ) +
    theme_minimal() +
    labs(
      title = "SHAP Beeswarm (Grouped Variables)",
      x = "Impact on Probability (SHAP Value)",
      y = NULL
    )

  ### SHAP Dependence Grid -----------------
  # Identify top variables that exist in the SHAP object
  top_vars <- shap_summary_table_grouped %>%
    dplyr::slice_max(mean_abs_shap, n = 6) %>%
    pull(Variable)
  top_vars_valid <- top_vars[top_vars %in% clean_label(colnames(shp_grouped$S))]
  
  test_original_X <- cell_test %>%
    select(all_of(names(modeling_data)[c(num_cols, cat_cols)])) %>%
    rename_with(clean_label)
  
  test_original_X <- test_original_X[, colnames(shap_grouped_matrix)]
  
  shp_grouped_original <- shapviz(shap_grouped_matrix, X = test_original_X)
  
  # Create a list of dependence plots
  dep_plots <- lapply(top_vars_valid, function(v) {
    sv_dependence(shp_grouped_original, v,
      color_var = "auto",
      alpha = 0.5,
      size = 1.2
    ) +
      theme_minimal(base_size = 10) +
      theme(
        axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1), # Slightly less angle
        axis.title.x = element_blank(), # REMOVE this to save space
        legend.position = "right",
        legend.title = element_text(size = 9),
        legend.text = element_text(size = 8),
        panel.spacing = unit(2, "lines")
      ) +
      labs(
        title = v,
        x = paste("Value of", v),
        y = "SHAP Value"
      )
  })

  final_dep_grid <- ggplot() +
    labs(title = "No dependence plots generated")

  # Combine the plots
  if (length(dep_plots) > 0) {
    final_dep_grid <- wrap_plots(dep_plots, ncol = 3) +
      plot_annotation(
        title = "SHAP Dependence Analysis: Feature Effects",
        subtitle = "Relationship between feature values and their impact on risk"
      )
  }

  ### Comparing plots ------------
  p_compare <- master_importance_table_original %>%
    # Dynamically find the importance column even if it was renamed
    rename(Importance = matches("Importance")) %>%
    slice_max(mean_abs_shap, n = 15) %>%
    mutate(
      gini_scaled = rescale(Importance),
      shap_scaled = rescale(mean_abs_shap)
    ) %>%
    pivot_longer(
      cols = c(gini_scaled, shap_scaled),
      names_to = "Metric", values_to = "Scaled_Importance"
    ) %>%
    ggplot(aes(x = reorder(Variable, Scaled_Importance), y = Scaled_Importance, fill = Metric)) +
    geom_col(position = "dodge") +
    coord_flip() +
    scale_fill_manual(
      values = c("#2c7bb6", "#d7191c"),
      labels = c("Gini/Gain (VI)", "Mean |SHAP|")
    ) +
    theme_minimal() +
    labs(
      title = "Feature Importance Consistency",
      subtitle = "Top 15 variables compared across VI and SHAP metrics",
      x = NULL, y = "Scaled Importance (0-1)"
    )

  ## 11.3 Decision Curve Analysis (DCA) -----------------
  if (mode == "classification") {
    dca_df <- final_preds %>%
      mutate(target_numeric = if_else(!!sym(target) == event_level, 1, 0)) %>%
      select(target_numeric, all_of(pred_col))
    dca_formula <- as.formula(paste("target_numeric ~", pred_col))
    dca_labels <- list()
    dca_labels[[pred_col]] <- "This ML Model"
    p_dca <- dca(dca_formula,
      data = dca_df,
      thresholds = seq(0, 0.50, by = 0.01),
      label = dca_labels
    ) %>%
      as_tibble() %>%
      ggplot(aes(x = threshold, y = net_benefit, color = label)) +
      geom_line(linewidth = 1) +
      theme_minimal() +
      labs(
        title = "Decision Curve Analysis: Clinical Utility",
        subtitle = "Model vs Default Strategies",
        x = "Threshold Probability",
        y = "Net Benefit",
        color = "Strategy"
      )
  }

  ## 11.4 SHAP waterfall plot ---------------
  ### Case Study Archetypes -----------------
  if (mode == "classification") {
    # Archetype 1: The "True Positive" (High confidence correct prediction)
    true_pos_id <- final_preds %>%
      mutate(row_id = row_number()) %>%
      filter(!!sym(target) == event_level & .pred_class_j == event_level) %>%
      slice_max(!!sym(pred_col), n = 1) %>%
      pull(row_id)

    # Archetype 2: The "False Positive" (High confidence error - The 'Near Miss')
    false_pos_id <- final_preds %>%
      mutate(row_id = row_number()) %>%
      filter(!!sym(target) == control_level & .pred_class_j == event_level) %>%
      slice_max(!!sym(pred_col), n = 1) %>%
      pull(row_id)

    # Archetype 3: The "Borderline Case" (Closest to the J-threshold)
    borderline_id <- final_preds %>%
      mutate(
        row_id = row_number(),
        dist_to_thresh = abs(!!sym(pred_col) - optimal_threshold)
      ) %>%
      slice_min(dist_to_thresh, n = 1) %>%
      pull(row_id)
    
    archetype_titles <- c("True Positive (High Risk)", 
                          "False Positive (False Alarm)", 
                          "Borderline (Tipping Point)")
    
  } else {
    # Archetype 1: The "Accurate High-Value" (High actual, high prediction)
    true_pos_id <- final_preds %>%
      mutate(row_id = row_number()) %>%
      filter(!!sym(target) > median(!!sym(target))) %>%
      slice_min(abs(!!sym(target) - .pred), n = 1) %>%
      pull(row_id)

    # Archetype 2: The "Major Over-prediction" (The 'False Alarm')
    false_pos_id <- final_preds %>%
      mutate(row_id = row_number(), error = .pred - !!sym(target)) %>%
      slice_max(error, n = 1) %>%
      pull(row_id)

    # Archetype 3: The "Typical Case" (Median target value)
    borderline_id <- final_preds %>%
      mutate(
        row_id = row_number(),
        dist_to_med = abs(!!sym(target) - median(!!sym(target)))
      ) %>%
      slice_min(dist_to_med, n = 1) %>%
      pull(row_id)
    
    archetype_titles <- c("Accurate High-Value", 
                          "Major Over-prediction", 
                          "Typical Case (Median)")
  }

  # Function to generate formatted waterfall plots
  plot_archetype <- function(id, title_suffix) {
    shp_clean <- shp
    colnames(shp_clean$S) <- clean_label(colnames(shp_clean$S))
    colnames(shp_clean$X) <- clean_label(colnames(shp_clean$X))

    # Determine labels based on mode
    target_val_raw <- final_preds[[target]][id]
    actual_val <- if(is.numeric(target_val_raw)) round(target_val_raw, 2) else target_val_raw
    pred_val <- if (mode == "classification") {
      round(final_preds[[pred_col]][id], 3)
    } else {
      round(final_preds[[".pred"]][id], 2)
    }
    label_type <- if (mode == "classification") "Prob:" else "Pred Value:"

    sv_waterfall(shp_clean, row_id = id, max_display = 10) +
      theme_minimal(base_size = 11) +
      labs(
        title = glue("Case {id}: {title_suffix}"),
        subtitle = glue("Actual: {actual_val} | {label_type} {pred_val}"),
        x = "SHAP Value (Contribution to Prediction)"
      )
  }

  ### Create the plots -----------
  p1 <- plot_archetype(true_pos_id, archetype_titles[1])
  p2 <- plot_archetype(false_pos_id, archetype_titles[2])
  p3 <- plot_archetype(borderline_id, archetype_titles[3])

  # Combine into a single figure
  case_study_panel <- (p1 / p2 / p3) +
    plot_annotation(
      title = "Clinical Case Studies: SHAP Explanations",
      theme = theme(plot.title = element_text(size = 18, face = "bold"))
    )

  # 12. Logistic or Linear Regression Specific Outputs --------------------------------
  if (model_type == "lr") {
    # 12.1 Bootstrap the model to get stable CIs ----------------
    set.seed(123)
    boot_lr_results <- bootstraps(cell_train, times = 100, strata = all_of(target)) %>%
      mutate(
        coef_info = map(splits, ~ {
          fit(final_wf, data = analysis(.x)) %>%
            extract_fit_parsnip() %>%
            tidy()
        })
      ) %>%
      unnest(coef_info)
    
    if(mode == "classification") {
    # 12.2.1 OR with CIs for classification --------------------
    forest_data <- boot_lr_results %>%
      filter(term != "(Intercept)") %>%
      group_by(term) %>%
      summarise(
        estimate_mean = mean(estimate),
        ci_low_beta = quantile(estimate, 0.025),
        ci_high_beta = quantile(estimate, 0.975),
        .groups = "drop"
      ) %>%
      mutate(
        OR = exp(estimate_mean),
        ci_low = exp(ci_low_beta),
        ci_high = exp(ci_high_beta),
        Variable = clean_label(term),
        signif = if_else(ci_low > 1 | ci_high < 1, "*", ""),
        or_label = glue("{round(OR, 2)} [{round(ci_low, 2)}-{round(ci_high, 2)}]{signif}"),
        label_with_std = reorder(Variable, estimate_mean)
      ) %>%
      slice_max(abs(estimate_mean), n = 20) # Keep top 20 predictors
    
    # 12.3.1 The Styled Forest Plot (Classification)-------------
    p_forest <- ggplot(forest_data, aes(x = OR, y = label_with_std)) +
      geom_vline(xintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.8) +
      geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, color = signif != ""), height = 0.2, linewidth = 0.9) +
      geom_point(aes(color = signif != ""), size = 3.5) +
      geom_text(aes(label = or_label), vjust = -1.2, size = 3.2, fontface = "bold") +
      scale_x_log10() +
      scale_color_manual(
        values = c("gray60", "#2c7bb6"),
        labels = c("Crossing Null (1.0)", "Stable (Excludes 1.0)"),
        name = "Bootstrap Stability"
      ) +
      labs(
        title = glue("Clinical Effect Sizes: {target}"),
        subtitle = "Plotted: Mean Bootstrap OR [95% CI] | Y-Axis: Top Predictors (Ordered by Estimate)",
        x = "Odds Ratio (Log Scale)",
        y = NULL
      ) +
      theme_minimal(base_size = 12) +
      theme(
        plot.title = element_text(face = "bold", size = 14),
        panel.grid.minor = element_blank(),
        legend.position = "bottom",
        axis.text.y = element_text(face = "bold")
      )
    }
    else {
    # 12.2.2 Coefficients for Linear Regression --------------------
    forest_data <- boot_lr_results %>%
      filter(term != "(Intercept)") %>%
      group_by(term) %>%
      summarise(
        estimate_mean = mean(estimate),
        ci_low = quantile(estimate, 0.025),
        ci_high = quantile(estimate, 0.975),
        .groups = "drop"
      ) %>%
      mutate(
        Variable = clean_label(term),
        # For Linear, check if CI crosses ZERO, not ONE
        signif = if_else(ci_low > 0 | ci_high < 0, "*", ""),
        label_text = glue("{round(estimate_mean, 2)} [{round(ci_low, 2)} to {round(ci_high, 2)}]{signif}"),
        label_with_std = reorder(Variable, estimate_mean)
      ) %>%
      slice_max(abs(estimate_mean), n = 20)
    
    # 12.3.2 The Styled Forest Plot (Regression) -------------
    p_forest <- ggplot(forest_data, aes(x = estimate_mean, y = label_with_std)) +
      # Reference line at 0 (No effect)
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray40", linewidth = 0.8) +
      geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, color = signif != ""), height = 0.2, linewidth = 0.9) +
      geom_point(aes(color = signif != ""), size = 3.5) +
      geom_text(aes(label = label_text), vjust = -1.2, size = 3.2, fontface = "bold") +
      # Use scale_x_continuous instead of scale_x_log10
      scale_x_continuous() + 
      scale_color_manual(
        values = c("gray60", "#d95f02"), # Changed color slightly to distinguish from classification
        labels = c("Crossing Null (0.0)", "Stable (Excludes 0.0)"),
        name = "Bootstrap Stability"
      ) +
      labs(
        title = glue("Linear Effect Sizes: {target}"),
        subtitle = "Plotted: Mean Bootstrap Coefficients [95% CI] | Y-Axis: Top Predictors",
        x = "Coefficient Estimate (Unit Change in Outcome)",
        y = NULL
      ) +
      theme_minimal(base_size = 12) +
      theme(
        plot.title = element_text(face = "bold", size = 14),
        panel.grid.minor = element_blank(),
        legend.position = "bottom",
        axis.text.y = element_text(face = "bold")
      )  
    }
  }
  
  # 13. Saving Everything ------------------
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  export_list <- list()
    
  add_if_exists <- function(name, label) {
    if (exists(name, envir = parent.frame(), inherits = FALSE)) {
      export_list[[label]] <<- get(name, envir = parent.frame())
    }
  }
  add_if_exists("feature_selection_results", "Feature Engineering")
  add_if_exists("best_params", "Best Hyperparameters")
  add_if_exists("threshold_summary","Threshold Summary")
  add_if_exists("diagnostic_summary", "Diagnostic Summary")
  add_if_exists("ci_values", "Metrics")
  add_if_exists("master_importance_table_dummy", "Var Imp Dummy")
  add_if_exists("master_importance_table_original", "Var Imp Original")
  add_if_exists("forest_data", "Penalized LR Results")
    
  if (length(export_list) > 0) {
      write.xlsx(export_list, file = file.path(output_dir, glue("{prefix}_Results.xlsx")))
      message("File saved successfully!")
  } else {
      warning("Export list is empty!")
  }
  
  lapply(
    list(
      list(obj = "final_dashboard", suf = "Diagnostic_Dashboard", w = 14, h = 12),
      list(obj = "p_vip_dummy", suf = "Importance_Dummy", w = 10, h = 8),
      list(obj = "p_vip_grouped", suf = "Importance_Original", w = 10, h = 8),
      list(obj = "p_shap", suf = "SHAP_Beeswarm_Dummy", w = 12, h = 9),
      list(obj = "p_shap_grouped", suf = "SHAP_Beeswarm_Original", w = 12, h = 9),
      list(obj = "p_compare", suf = "Importance_Comparison", w = 10, h = 8),
      list(obj = "final_dep_grid", suf = "SHAP_Dependence_Grid", w = 12, h = 15),
      list(obj = "p_dca", suf = "DCA_Clinical_Utility", w = 8, h = 7),
      list(obj = "case_study_panel", suf = "SHAP_Waterfall_Archetypes", w = 12, h = 18),
      list(obj = "p1", suf = "The_True_Positive", w = 8, h = 7),
      list(obj = "p2", suf = "The_False_Positive", w = 8, h = 7),
      list(obj = "p3", suf = "The_Borderline_Case", w = 8, h = 7),
      list(obj = "p_forest", suf = "LR_Forest_Plot", w = 10, h = 10)
    ),
    function(conf) {
      if (exists(conf$obj, where = -1, inherits = TRUE)) {
        save_path <- file.path(output_dir, 
                               glue("{prefix}_{conf$suf}.png"))
        ggsave(filename = save_path, 
               plot = get(conf$obj, inherits = TRUE), 
               width = conf$w, 
               height = conf$h, 
               dpi = 300, 
               bg = "white")
      }
    }
  )
  readr::write_rds(final_fitted_wf, file.path(output_dir, paste0(prefix, "_final_model.rds")), compress = "gz")
  readr::write_rds(race_results, file.path(output_dir, paste0(prefix, "_tuning_resamples.rds")), compress = "gz")
  return(list(
    metrics = ci_values,
    diagnostic = diagnostic_summary,
    importance_dummy = master_importance_table_dummy,
    importance_orig = master_importance_table_original,
    predictions = final_preds,
    roc_obj = if(mode == "classification") roc_obj_j else NULL,
    fit = final_fit,
    final_model = final_fitted_wf,
    mode = mode,    
    target_var = target,      
    boot_dist = boot_metrics,
    cv_results = cv_fold_data
  ))

}
