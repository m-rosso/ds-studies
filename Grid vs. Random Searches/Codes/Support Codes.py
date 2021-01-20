corrected_model_assessment = {}

for e in model_assessment.keys():
    if (model_assessment[e]['method'] == 'logistic_regression') | ((model_assessment[e]['method'] == 'GBM') & (model_assessment[e]['random_search'] == False)):
        corrected_model_assessment[e] = {
                "store_id": model_assessment[e]["store_id"],
                "n_orders_train": model_assessment[e]["n_orders_train"],
                "n_orders_test": model_assessment[e]["n_orders_test"],
                "n_vars": model_assessment[e]["n_vars"],
                "first_date_train": model_assessment[e]["first_date_train"],
                "last_date_train": model_assessment[e]["last_date_train"],
                "first_date_test": model_assessment[e]["first_date_test"],
                "last_date_test": model_assessment[e]["last_date_test"],
                "avg_order_amount_train": model_assessment[e]["avg_order_amount_train"],
                "avg_order_amount_test": model_assessment[e]["avg_order_amount_test"],
                "log_transform": model_assessment[e]["log_transform"],
                "standardize": model_assessment[e]["standardize"],
                "method": model_assessment[e]["method"],
                "random_search": model_assessment[e]["random_search"],
                "n_samples": model_assessment[e]["n_samples"],
                "best_param": model_assessment[e]["best_param"],
                "test_roc_auc": model_assessment[e]["test_brier"],
                "test_prec_avg": model_assessment[e]["test_roc_auc"],
                "test_brier": model_assessment[e]["test_prec_avg"],
                "running_time": model_assessment[e]["running_time"]
        }
    
    elif ((model_assessment[e]['method'] == 'GBM') & (model_assessment[e]['random_search'] == True)):
        corrected_model_assessment[e] = {
                "store_id": model_assessment[e]["store_id"],
                "n_orders_train": model_assessment[e]["n_orders_train"],
                "n_orders_test": model_assessment[e]["n_orders_test"],
                "n_vars": model_assessment[e]["n_vars"],
                "first_date_train": model_assessment[e]["first_date_train"],
                "last_date_train": model_assessment[e]["last_date_train"],
                "first_date_test": model_assessment[e]["first_date_test"],
                "last_date_test": model_assessment[e]["last_date_test"],
                "avg_order_amount_train": model_assessment[e]["avg_order_amount_train"],
                "avg_order_amount_test": model_assessment[e]["avg_order_amount_test"],
                "log_transform": model_assessment[e]["log_transform"],
                "standardize": model_assessment[e]["standardize"],
                "method": model_assessment[e]["method"],
                "random_search": model_assessment[e]["random_search"],
                "n_samples": model_assessment[e]["n_samples"],
                "best_param": model_assessment[e]["best_param"],
                "test_roc_auc": model_assessment[e]["test_roc_auc"],
                "test_prec_avg": model_assessment[e]["test_prec_avg"],
                "test_brier": model_assessment[e]["test_brier"],
                "running_time": model_assessment[e]["running_time"]
        }

with open('Datasets/model_assessment.json', 'w') as json_file:
    json.dump(corrected_model_assessment, json_file, indent=2)
