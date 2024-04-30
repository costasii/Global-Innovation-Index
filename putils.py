import typing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB



def show_prep_graphs(df):
    fig = plt.figure(figsize=(8, 3))

    ax = fig.add_subplot(121)
    ax.plot(df.isna().sum().sort_values())
    ax.set_title("Number of missing values for each column\n(ascending order)")
    ax.set_ylabel("Num of NaNs")
    ax.set_xticks([])

    ax = fig.add_subplot(122)
    dfg = df.reset_index()[['DATE', 'Global Innovation Index']].groupby(
        ['DATE']).mean()
    ax.plot(dfg)
    ax.set_title("Global Innovation Index\n(Yearly mean)")
    ax.set_ylabel("Innovation Index Average")
    ax.set_xlabel("Year")
    ax.set_xticks(dfg.index, dfg.index, rotation=45)

    plt.tight_layout()
    plt.show()


def min_max_normalization(df):

    return (df-df.min())/(df.max()-df.min())  # MinMax Normalization


def mean_normalization(df):
    return (df-df.mean())/df.std()  # Mean Normalization


def get_most_relevant_features_rf(X_train, X_test, y_train, plot=True):

    print("[best_f_rf] Getting best features...")
    X = X_train
    y = y_train
    n_features = X_train.shape[1]

    embeded_rf_selector = SelectFromModel(RandomForestRegressor(
        n_estimators=500, random_state=42), max_features=n_features)
    embeded_rf_selector.fit(X, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()
    print(
        f"[best_f_rf] {str(len(embeded_rf_feature))} features are significant enough according to RandomForestClassifier.")

    # if len(embeded_rf_feature) <20:
    print(f'[best_f_rf] {embeded_rf_feature}')

    if plot == True:
        plot_feature_importance(
            embeded_rf_selector.estimator_.feature_importances_, X.columns, "RANDOM FOREST")

    # return embeded_rf_feature
    # select only important columns

    X_train_reduced = X_train[embeded_rf_feature]
    X_test_reduced = X_test[embeded_rf_feature]

    return X_train_reduced, X_test_reduced


def plot_feature_importance(importance, names, model_type):

    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names,
            'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def plot_feature_variance(variance, names, threshold=0.2):

    # Create arrays from feature importance and feature names
    feature_variance = np.array(variance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names,
            'feature_variance': feature_variance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_variance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    graph = sns.barplot(x=fi_df['feature_variance'], y=fi_df['feature_names'])
    # graph.axvline(threshold)
    # Add chart labels
    plt.title('FEATURE VARIANCE')
    # plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def plot_reg_prediction_fit(y_true, y_pred, dataset_name, classifer_name):

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, c='crimson')
    # plt.yscale('log')
    # plt.xscale('log')

    p1 = max(max(y_pred), max(y_true))
    p2 = min(min(y_pred), min(y_true))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.title(
        f'Pred. vs True values - [{classifer_name}, {dataset_name}]', fontsize=10, fontweight="bold")
    plt.axis('equal')
    # plt.savefig(f"./figures/{dataset_name}_predvstrue_{classifer_name}.png")
    plt.show()


def reduce_pca_components(X_train, X_test, plot=True):

    pca = PCA(n_components=0.85)
    pca.fit(X_train)
    print("Cumulative Variances (Percentage):")
    print(np.cumsum(pca.explained_variance_ratio_ * 100))
    components = len(pca.explained_variance_ratio_)
    print(f'Number of components: {components}')

    if plot == True:
        # Make the scree plot
        plt.plot(range(1, components + 1),
                 np.cumsum(pca.explained_variance_ratio_ * 100))
        plt.xlabel("Number of components")
        plt.ylabel("Explained variance (%)")

    print('Top 4 most important features in each component')
    print('===============================================')
    pca_components = abs(pca.components_)
    for row in range(pca_components.shape[0]):
        # get the indices of the top 4 values in each row
        temp = np.argpartition(-(pca_components[row]), 4)

        # sort the indices in descending order
        indices = temp[np.argsort((-pca_components[row])[temp])][:4]

        # print the top 4 feature names
        print(f'Component {row}: {X_train.columns[indices].to_list()}')

    # reduce the df
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(X_train_pca.shape)

    return X_train_pca, X_test_pca


def apply_variance_threshold_feature_selection(X_train: pd.DataFrame,
                                               X_valid: pd.DataFrame,
                                               threshold: float,
                                               ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Apply the VarianceTrheshold feature selection method to X_train and 
    X_vaild.

    Parameters
    --------
    threshold: float, threshold to be used for VarianceThreshold feature selector

    Returns
    --------
    X_train_sel: dataframe, selected columns from X_train
    X_valid_sel: dataframe, selected columns from X_valid

    """
    X_train_sel = None
    X_valid_sel = None

    # print(X_train.columns)

    # YOUR CODE HERE
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold)
    selector.fit(X_train)
    feature_index = selector.get_support(indices=True)
    feature_names = [X_train.columns[i] for i in feature_index]

    X_train_sel = pd.DataFrame(selector.transform(X_train))
    X_train_sel.columns = feature_names
    X_train_sel.set_index(X_train.index, inplace=True)

    # replicate for test
    X_valid_sel = X_valid[X_train_sel.columns]

    # print(X_train.head(5))
    # print(X_train_sel.columns)
    # print(X_valid_sel)
    # print(X_train_sel)

    return X_train_sel, X_valid_sel


# The following functions help with training the models and finding the optimal one.

def get_suitable_ml_methods() -> typing.List[typing.Type[sklearn.base.BaseEstimator]]:
    """ 
    Returns
    --------
    suitable_methods: list of types (classes) for suitable sklearn machine learning models.
    """
    suitable_methods = []

    suitable_methods.append(sklearn.linear_model.Ridge)
    suitable_methods.append(sklearn.linear_model.Lasso)
    suitable_methods.append(sklearn.linear_model.ElasticNet)
    suitable_methods.append(sklearn.linear_model.BayesianRidge)
    suitable_methods.append(sklearn.svm.SVR)
    suitable_methods.append(sklearn.linear_model.LinearRegression)

    
    return suitable_methods

def print_selection(selected: list, sel_type:str='methods'):
    print(f"Identified ML {sel_type}:\n===================\n"+ 
           '\n'.join([cur_sel.__name__ for cur_sel in selected 
                 if hasattr(cur_sel, '__name__')]))

def train_model(X_train:pd.DataFrame, y_train:pd.Series, 
                model_class: typing.Type[sklearn.base.BaseEstimator]
                ) -> sklearn.base.BaseEstimator:

    trained_model = model_class()  # instanciate sklearn model to be trained

    # train the model:

    trained_model.fit(X_train, y_train)

    return trained_model

def get_suitable_metrics() -> typing.List[typing.Callable]:
    """ 
    Returns
    --------
    suitable_methods: list of types (classes) for suitable sklearn machinelearning models.)
    """
    suitable_metrics = []

    suitable_metrics.append(sklearn.metrics.mean_squared_error)
    suitable_metrics.append(sklearn.metrics.r2_score)
    suitable_metrics.append(sklearn.metrics.max_error)
    
    return suitable_metrics

def predict_and_eval(trained_model: sklearn.base.BaseEstimator, 
                     metric_func: typing.Callable, 
                     X_valid: pd.DataFrame, 
                     y_true: pd.Series) -> typing.Tuple[np.ndarray, float]:
    """ 
    Returns
    --------
    y_pred: np.ndarray of predictions for weekly_infections from X_valid features
    metric_result: float, the result of metric_func for the generated predictions y_pred
                   compared to true values y_true.
    """
    y_pred = None
    metric_result = None

    y_pred = trained_model.predict(X_valid)
    
    metric_result = metric_func(y_true, y_pred)
 
    return y_pred, metric_result

def compare_methods(method_classes: typing.List[typing.Type[sklearn.base.BaseEstimator]],
                    metric_functions: typing.List[typing.Callable],
                    X_train: pd.DataFrame, y_train:pd.Series,
                    X_valid: pd.DataFrame, y_valid: pd.Series
                    ) -> typing.Dict[str, float]:

    scores = {} # dict of metric name -> metric value/score
    
    for method in method_classes:
        for metric in metric_functions:
            trained_model = train_model(X_train, y_train, method)
            y_pred = trained_model.predict(X_valid)
            metric_result = metric(y_valid, y_pred)
            method_name = method.__name__
            metric_name = metric.__name__
            name = str(method_name) + " " + "------" + " " + str(metric_name)

            scores[name] = metric_result
            
    return scores

def print_scores(method_classes: typing.List[typing.Type[sklearn.base.BaseEstimator]],
                metric_functions: typing.List[typing.Callable],
                scores: typing.Dict[str, float]):
    print("Scores:\n=======")

    header = []
    for method in method_classes:
        header.append(method.__name__)

    first_col = []
    for metric in metric_functions:
        first_col.append(metric.__name__)

    score_values = []
    for key, value in scores.items():
        score_values.append(value)
    
    score_values = np.reshape(score_values, (len(method_classes), len(metric_functions)))
    
    print(pd.DataFrame(score_values, header, first_col))
