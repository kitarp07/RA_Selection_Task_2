from src.modelling import build_pipeline, train_and_evaluate_logistic_regression_model, get_top_features, train_final_rf_model, train_and_evaluate_rf_model, train_and_evaluate_svm_model
from src.preprocessing import ( load_data,remove_almost_constant_features, remove_features_with_na_rows, remove_highly_correlated_features, 
                               replace_inf_with_nan, fill_cols_with_na_val, scale_values, select_k_best, preprocess_data, load_blinded_data )


def main():
    path_to_train_data = "./data/train_set.csv"
    path_to_test_data = "./data/test_set.csv"
    path_to_blinded_data = "./data/blinded_test_set.csv"


    X_train, y_train, train_id = load_data(path_to_train_data)
    X_test, y_test, test_id = load_data(path_to_test_data)
    X_blinded_test, blind_test_id = load_blinded_data(path_to_blinded_data)
    
    # preprocess data
    
    X_train_preprocessed, X_test_filtered, X_blinded_test_filtered = preprocess_data(X_train, X_test, X_blinded_test)
    
    #scale values
    X_train_scaled, X_test_scaled, X_blinded_test_scaled = scale_values(X_train_preprocessed, X_test_filtered, X_blinded_test_filtered)
    
    # select top k features
    X_train_selected, X_test_selected, X_blinded_test_selected, selected_features = select_k_best(X_train_scaled, y_train, X_test_scaled, X_blinded_test_scaled, k=60)
    
    #train and evaluate using logistic regression    
    pipeline = build_pipeline()
    pipeline, selected_columns= train_and_evaluate_logistic_regression_model(pipeline, X_train_selected, y_train, X_test_selected, y_test, X_blinded_test_selected, train_id,test_id,blind_test_id)
    print("Logistic Regression evaluation complete")
    
    #train and evaluate using random forest
    metrics = train_and_evaluate_rf_model(X_train_preprocessed, y_train, X_test_filtered, y_test, X_blinded_test_filtered, train_id,test_id,blind_test_id)
    print("Random forest evaluation complete")

    
    #train and evaluate using svm
    
    x_train_svm = X_train_selected[selected_columns]
    x_test_svm = X_test_selected[selected_columns]
    X_blinded_test_svm = X_blinded_test_selected[selected_columns]

    metrics = train_and_evaluate_svm_model(x_train_svm, y_train, x_test_svm, y_test, X_blinded_test_svm, train_id,test_id,blind_test_id)
    print("SVM evaluation complete")
    
if __name__ == "__main__":
    main()


    
    
    