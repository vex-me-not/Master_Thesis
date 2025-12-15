import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import optuna
import lightgbm as lgb
import warnings

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from optuna.pruners import MedianPruner
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import r_regression
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder,RobustScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score,f1_score, precision_score, recall_score, confusion_matrix, average_precision_score
from scipy import stats


warnings.filterwarnings('ignore')

class IO:
    """
    This class implements the saving and the loading of the models. It needs a models directory in the constructor to work
    So, each instance of IO class coresponds to one specific directory.
    If you want to load or to save from another directory, you need to create a new instance
    """
    def __init__(self,models_dir):
        if models_dir is None:
            raise ValueError("Directory of models must be given!")

        self.models_dir=Path(models_dir)
        os.makedirs(self.models_dir,exist_ok=True)

    def __gen_filename(self,name,suffix):
        parts=[name]
        if suffix:
            parts.append(suffix)

        return self.models_dir/f"{'_'.join(parts)}.pkl"
    
    # method that saves a model adding a suffix suf to its name
    def save(self,model,name,suf=''):
        joblib.dump(model,self.__gen_filename(name,suffix=suf))

    # method
    def load(self,name,suf=''):
        return joblib.load(self.__gen_filename(name,suffix=suf))

class EarlyStopping:
    """
    This class implements the early stopping functionality in the optimize function of the optuna studies. The idea is that after patience
    trials have passed when no improvement has occured, the study will be stopped.
    """
    # the constructor of the class 
    def __init__(self, patience):
        self.patience = patience
        self.best = None
        self.no_impr_count = 0

    # the main mathod that gives this class the desired functionality. The trial parameter is needed because of the callbacks parameter in create study.
    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        if (self.best is None) or (study.best_value > self.best):
            self.best = study.best_value
            self.no_impr_count=0
        else:
            self.no_impr_count += 1

        if (self.no_impr_count >= self.patience):
            print(f"EARLY STOPPING: No improvement after {self.patience} trials!")
            study.stop()
            print(f"--> Best trial is {study.best_trial.number} with value: {study.best_trial.value} and parameters: {study.best_trial.params}\n")

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    This class implements the columns selection functionality. In simpler terms, with this class we are able to include the 
    feature selection phase of our model to a pipeline
    """    
    # the constructor of the class
    def __init__(self, columns):
        self.columns=columns

    # the fit method of the class needed for the pipeline
    def fit(self, x, y=None):
        return self

    # the transform method of the class, needed for the pipeline
    def transform(self, x):
        return x[self.columns]

class rnCV():

    """
    This class implements the repeated nested cross validation pipeline. The default values of r,n,k are those of the exercise but all can be
    changed through the constructor. The run_rnCV method is used to run the pipeline on the give data_df dataset, whilst the result summary is
    used to get the results of the pipeline in a dataframe.
    """
    # the constructor of the class. Notice that we don't have to give x and y, just the dataset
    def __init__(self,data_df,estimators,params,r=10,n=5,k=3,random_state=42):
        self.R=r
        self.N=n
        self.K=k
        self.x, self.y=keep_features(data_df=data_df,target='diagnosis',to_drop=['id'])
        self.y= encode(self.y)
        self.estimators=estimators
        self.params=params
        self.random_state=random_state
        self.results={estim:[] for estim in estimators.keys()}

    # method used to calculate the various metrics required.
    def _compute_metrics(self,y_true,y_pred,y_prob):
        tn, fp, fn, tp=confusion_matrix(y_true, y_pred).ravel()
        specificity=tn / (tn + fp)
        npv=tn / (tn + fn)

        metrics={
            'MCC': matthews_corrcoef(y_true, y_pred),
            'ROC_AUC': roc_auc_score(y_true, y_prob),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Specificity': specificity,
            'NPV': npv,
            'PR_AUC': average_precision_score(y_true, y_prob)
        }

        return metrics
    
    def results_summary(self):
        summary = {}
        
        for model_name, all_metrics in self.results.items():
            df=pd.DataFrame(all_metrics)
            summary[model_name]=df.median().to_dict()
    
        summary_t=pd.DataFrame(summary).T
        
        return summary_t

    # the inner loop
    def _tune_model(self,x_train,y_train,model_name):
        np.random.seed(42)
        # per optuna's documentation, we create the objective function that will be maximized
        def objective(trial):
            estimator=self.estimators[model_name]
            params=self.params[model_name](trial)
            model=estimator(**params)

            # we create k-folds whilst keeping the imbalances with StratifiedKFold
            strat_kfold=StratifiedKFold(n_splits=self.K, shuffle=True, random_state=self.random_state)
            scores=[]

            for fold_idx, (k_train_idx, k_val_idx) in enumerate(strat_kfold.split(x_train, y_train)):
                x_k_train, x_val = x_train.iloc[k_train_idx], x_train.iloc[k_val_idx]
                y_k_train, y_val = y_train.iloc[k_train_idx], y_train.iloc[k_val_idx]
                
                scaler=RobustScaler()
                trans=PowerTransformer(method='yeo-johnson')
                imp=IterativeImputer(random_state=42)

                scaler.fit(x_k_train)
                trans.fit(x_k_train)
                imp.fit(x_k_train)
                # we impute and scale here as to avoid data leakage
                x_k_train=impute(x_k_train,imp=imp)
                x_k_train=scale_data(x_k_train,scaler=scaler,transformer=trans)
                
                x_val=impute(x_val,imp=imp)
                x_val=scale_data(x_val,scaler=scaler,transformer=trans)

                # we fit and we predict
                model.fit(x_k_train, y_k_train)
                preds=model.predict(x_val)

                # intermediate results, used in pruning
                score=balanced_accuracy_score(y_val, preds) 
                scores.append(score)

                # report intermediate result for pruning
                trial.report(score, step=fold_idx)
                
                # check if the trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)        
        

        study=optuna.create_study(direction='maximize',study_name=model_name)
        study.optimize(objective, n_trials=60,timeout=180.0,callbacks=[EarlyStopping(patience=10)])
        
        return study.best_params        

    def run_rnCV(self):
        np.random.seed(self.random_state)
        
        # we run for all the repetitions
        for r in range(self.R):
            print(f"------ Repetition {r+1}/{self.R} ------\n")

            state=self.random_state + r
            strat_kfold=StratifiedKFold(n_splits=self.N, shuffle=True, random_state=state)

            # we perform the outer loop for each model
            for model_name in self.estimators.keys():
                # the outer loop
                for n_train_idx, n_test_idx in strat_kfold.split(self.x, self.y):
                    x_n_train, x_n_test=self.x.iloc[n_train_idx], self.x.iloc[n_test_idx]
                    y_n_train, y_n_test=self.y.iloc[n_train_idx], self.y.iloc[n_test_idx]

                    scaler=RobustScaler()
                    trans=PowerTransformer(method='yeo-johnson')
                    imp=IterativeImputer(random_state=42)

                    scaler.fit(x_n_train)
                    trans.fit(x_n_train)
                    imp.fit(x_n_train)
                    # we run the inner loops
                    best_params=self._tune_model(x_train=x_n_train,y_train= y_n_train,model_name=model_name)

                    model=self.estimators[model_name](**best_params)

                    # we impute and scale here, as to avoid data leakage
                    x_n_train=impute(x_n_train,imp=imp)
                    x_n_train=scale_data(x_n_train,scaler=scaler,transformer=trans)
                    
                    x_n_test=impute(x_n_test,imp=imp)
                    x_n_test=scale_data(x_n_test,scaler=scaler,transformer=trans)
                    
                    # we fit and we predict
                    model.fit(x_n_train, y_n_train)

                    preds=model.predict(x_n_test)
                    pred_probs=model.predict_proba(x_n_test)[:, 1]

                    # we compute the required metrics
                    metrics=self._compute_metrics(y_true=y_n_test,y_pred=preds,y_prob=pred_probs)
                    self.results[model_name].append(metrics)

        return self.results

    # method used to find the optimal set of hyperparameters for the winner model
    def tune_winner(self,winner):
        np.random.seed(42)
        # per optuna's documentation
        def objective(trial):
            estimator=self.estimators[winner]
            params=self.params[winner](trial)
            model=estimator(**params)

            strat_kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores=[]

            x_tune=self.x
            y_tune=self.y
            
            for fold_idx, (tune_train_idx, tune_val_idx) in enumerate(strat_kfold.split(x_tune, y_tune)):
                x_tune_train, x_tune_val= x_tune.iloc[tune_train_idx], x_tune.iloc[tune_val_idx]
                y_tune_train, y_tune_val= y_tune.iloc[tune_train_idx], y_tune.iloc[tune_val_idx]

                scaler=RobustScaler()
                trans=PowerTransformer(method='yeo-johnson')
                imp=IterativeImputer(random_state=42)

                scaler.fit(x_tune_train)
                trans.fit(x_tune_train)
                imp.fit(x_tune_train)

                # we impute and scale here, as to avoid data leakage                    
                x_tune_train=impute(x_tune_train,imp=imp)
                x_tune_train=scale_data(x_tune_train,scaler=scaler,transformer=trans)
                    
                x_tune_val=impute(x_tune_val,imp=imp)
                x_tune_val=scale_data(x_tune_val,scaler=scaler,transformer=trans)

                # we fit and we predict
                model.fit(x_tune_train, y_tune_train)
                preds=model.predict(x_tune_val)

                # intermediate results, used in pruning                
                score=balanced_accuracy_score(y_tune_val, preds)
                scores.append(score)

                # report intermediate result for pruning
                trial.report(score, step=fold_idx)
                    
                # check if the trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)        
            

        study=optuna.create_study(direction='maximize',study_name="Winner:"+winner)
        study.optimize(objective, n_trials=60,timeout=180.0,callbacks=[EarlyStopping(patience=10)])
            
        return study.best_params


# method used to run the repeated nested cross validation pipeline
def perform_rnCV(path):
    # we load the given dataset into a data frame
    df=pd.read_csv(path)

    # we define the estimators to be used
    estimators = {
        'LogisticRegression': LogisticRegression,
        'GaussianNB': GaussianNB,
        'LDA': LinearDiscriminantAnalysis,
        'SVM': SVC,
        'RandomForest': RandomForestClassifier,
        'LightGBM': lgb.LGBMClassifier
    }

    # we define the hyperparameter spaces for the aforementioned estimators
    param_spaces = {
        'LogisticRegression': lambda trial: {
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['saga']),
            'C': trial.suggest_float('C', 1e-3, 1e0, log=True),
            'l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1),
            'random_state': trial.suggest_categorical('random_state', [42])
        },
        'GaussianNB': lambda trial: {'var_smoothing': trial.suggest_float('var_smoothing', 1e-2, 1e-1, log=True)},
        'LDA': lambda trial: {'solver':trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
                              'tol':trial.suggest_float('tol', 5*1e-2, 1e-1, log=True)},
        'SVM': lambda trial: {
            'C': trial.suggest_float('C', 5*1e-2, 1e2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'probability': trial.suggest_categorical('probability', [True]),
            'random_state': trial.suggest_categorical('random_state', [42])            
        },
        'RandomForest': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state': trial.suggest_categorical('random_state', [42])
        },
        'LightGBM': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'random_state': trial.suggest_categorical('random_state', [42]),
            'verbosity': trial.suggest_categorical('verbosity', [-1])
        }
    }

    # we initialize a rnCV class instance and we run_rnCV
    rncv=rnCV(data_df=df, estimators=estimators,params=param_spaces, r=10, n=5, k=3, random_state=42)
    results=rncv.run_rnCV()

    # we summarize and save the results in a dataframe
    summary=rncv.results_summary()
    summary.to_csv('../data/rncv_summary_results.csv')

    print("Summary:\n", summary)

# method used to clean the data. The title is a tad misleading, as there are also slivers of data exploration as well
def clean_data(data:pd.DataFrame):
    df=data

    # we get some general info about our data
    general_info(df)

    # we find the columns that have missing values and we count the total missing values
    nan_columns=df.columns[df.isna().any()].tolist()
    total_nan=(df.isna().sum()).sum()

    # we find the columns that are numeric and we drop the column id from them
    df_numeric=df.select_dtypes(include=[float, int])
    df_numeric=df_numeric.drop(columns=['id'])

    # we print some general info before the imputing
    print(f'Our data consists of {df.shape[1]} columns and {df.shape[0]} entries')
    print(f'We have {df_numeric.shape[1]} numeric columns. These are {list(df_numeric.columns)}')
    print(f'{len(nan_columns)} columns have missing values. These columns are : {nan_columns}')
    print(f'In total we have {total_nan} missing values')

    # we impute our data, thus replacing all NA values
    df=impute(data_df=df,imp=None,lacks=True)

    # we check that the imputing has indeed worked
    nan_columns=df.columns[df.isna().any()].tolist()
    print(f'We now have {len(nan_columns)} columns with missing values. These columns are : {nan_columns}')

    # we encode our two classes (Malignan -> 1, Benign -> 0)    
    df=encode(data_df=df,target='diagnosis')

    # we remove the duplicates
    df=remove_duplicates(df)

    # we check for outliers
    check_for_outliers(df)

    # but we WON'T remove them, we will just have to treat them
    print("We WON'T remove the outliers.")

    return df


# method that returns some genral info about our dataset
def general_info(data_df: pd.DataFrame):
    print(f'Shape of dataset: {data_df.shape} ({data_df.shape[0]} entries and {data_df.shape[1]} columns)')
    print(f'Data type of the {data_df.shape[1]} columns\n {data_df.dtypes}')

# method used to impute, a.k.a replace NAs
def impute(data_df: pd.DataFrame,imp,lacks=False):
    df=data_df

    # we find the numeric columns of the dataframe
    df_numeric=df.select_dtypes(include=[float, int])

    # we drop id column    
    if 'id' in df_numeric.columns:
        df_numeric=df_numeric.drop(columns=['id'])

    # we use the IterativeImputer to impute
    if lacks is True:
        imp=IterativeImputer(random_state=42)
        df[df_numeric.columns]=imp.fit_transform(df_numeric)
    else:
        df[df_numeric.columns]=imp.transform(df_numeric)

    return df

def encode(data_df: pd.DataFrame,target='diagnosis'):
    """
    Method used to encode the entries of the column 'diagnosis'
    Malignant --> 1
    Benign --> 0
    """
    df=data_df
    
    # we use the LabelEncoder to encode the two classes
    df[target]=LabelEncoder().fit_transform(df[target]) # M->1, B->0

    return df
    

def remove_duplicates(data_df: pd.DataFrame):
    """
    We use this function to find and remove any potential duplicates
    """
    
    df=data_df
   
    shape_before=df.shape
    df.drop_duplicates()
    shape_after=df.shape

    # we check for diffences in the shape. If there are, then the dataset used to have duplicate values
    if (shape_before[0] != shape_after[0]):
        print("Before removal of duplicates",shape_before)
        print("After removal of duplicates",shape_after)
    else:
        print("No duplicates in the set")
    
    return df

# method used to check for outliers and predict what our data would look like without them
def check_for_outliers(data_df: pd.DataFrame):

    df=data_df
   
    shape_before=df.shape
    
    # we drop all the outliers
    no_outliers=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    shape_after=no_outliers.shape

    # If there are differences in shape, then our dataset had outliers. We suggest what would happen if we were to remove them
    if (shape_before[0] != shape_after[0]):
        removed=shape_before[0]-shape_after[0]
        print("Before removal of outliers",shape_before)
        class_imbalance(df)
        print("After removal of outliers",shape_after)
        class_imbalance(no_outliers)
        print(f"We could remove {removed} entries ({(removed/shape_before[0])*100:.2f}% of total entries)")

    else:
        print("No outliers in the set")

# method used to explore the class imbalance of our dataset
def class_imbalance(data_df: pd.DataFrame,field='diagnosis'):
    df=data_df
    order=[0,1] # the order that we want to present our classes

    # we find how many entries per class
    entries=df[field].value_counts().reindex(order)
    print(f'Absolute frequencies of field "{field}"')
    print(entries)

    # the same as above, but in percentage
    fractions=df[field].value_counts(normalize=True)
    print(f'Percentage of each class of field "{field}"')
    print(fractions)

# method used to get the target of each dataset. Raises a ValueError if given target does not exist
def get_Y(data_df: pd.DataFrame,target='diagnosis'):
    if target not in data_df.columns:
        raise ValueError("Please give a valid target!")
    
    return pd.DataFrame(data_df[target])

# method used to keep the features and the target. Return x and y
def keep_features(data_df: pd.DataFrame,target='diagnosis',to_drop=None):
    tdrp=[]
    
    # we update our columns to be dropped with the list given
    if to_drop is not None:
        tdrp = [col for col in to_drop if col in data_df.columns]
    
    tdrp.append(target) # we add the target to the columns to be dropped
    Y=get_Y(data_df=data_df,target=target)
    x=data_df.drop(tdrp,axis=1)

    return x,Y

# method used to calculate the correlation between each feature and the target. We use thres to filter the non-significant
def corr_between_target(data_df: pd.DataFrame,target='diagnosis',thres=0.1):
    x,Y=keep_features(data_df=data_df,target=target,to_drop=['id']) # we get the feature and the target
    corr=pd.Series(r_regression(x,Y),index=x.columns) # we calculate the correlation of each feature
    selected=corr[corr.abs() >= thres].index.tolist() # we filter based on thres

    print("We could keep only the most correlated features:")
    print(selected)

    print(f'If we do, we will go from {x.shape[1]} features to {len(selected)} features,that will be the most correlated')

    print("Returning the features that could be kept")
    viz_corr_between_target(corr=corr,target='diagnosis') # we visualize the above results 


    return selected

# method used to calculate the correlation between the features. We use thres to keep only the most correlated
def corr_between_features(data_df: pd.DataFrame,target='diagnosis',to_drop=['diagnosis','id'],thres=0.8):
    df=data_df

    # we find our features
    feats=df.drop(columns=to_drop).columns.to_list() 

    # we calculate the pair-wise correlation between each possible combination
    corr_matrix=df[feats].corr(method='pearson')

    viz_corr_between_features(corr_matrix=corr_matrix) # we visualize that with the help of heatmap
    
    # we keep only the upper half of the correlation matrix, as the lower half is symmetrical
    corr_pairs=corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False)

    # we find the higly correlated pairs
    high_pairs=corr_pairs[corr_pairs>=thres]
    print('Pairs of high correlation')
    print(high_pairs)

    high_to_drop=set()

    # if we were to drop, we would drop the second part of the pair
    for high_pair in high_pairs.index:
        high_to_drop.add(high_pair[1])
    
    print("We could remove these features:")
    print(high_to_drop)

    high_selected=df.drop(columns=high_to_drop)
    print(f"If we do, we will go from {len(feats)} features to {len(high_selected.columns)} features")
    print('Returning the features that could be ignored')

    return high_to_drop


# method used to visualize the correlation between features.
def viz_corr_between_features(corr_matrix):
    
    # we visualize with the help of heatmap
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, cmap=sns.cubehelix_palette(as_cmap=True), center=0,linewidth=.5)
    
    plt.title('Heatmap of Correlation between Features')
    plt.show()


# method used to visualizr the correlation between features and target
def viz_corr_between_target(corr:pd.Series,target='diagnosis'):
    plt.figure(figsize=(10,6))
    
    corr.sort_values().plot(kind='barh',color='salmon')
    
    plt.title(f"Feature Correlations with {target}")
    plt.xlabel("Pearson's Correlation Coefficient")
    plt.ylabel("Feature")
    
    plt.axvline(0, color='black', linestyle='--')
    
    plt.show()

# method used to visualize the distribution of the features through boxplots. It plots everything into one plot
def boxpolt_distro(data_df: pd.DataFrame,to_drop=['diagnosis','id']):
    df=data_df

    feats=df.drop(columns=to_drop).columns.to_list() # our features

    plt.figure(figsize=(12,30))

    # we iterate over the features
    for i,feat in enumerate(feats,1):
        plt.subplot(10, 3, i) # the specific position of the feature in plot "grid"
        sns.boxplot(x=df[feat], color='salmon',flierprops={"marker":"x"}) # the boxplot with the distribution
        plt.tight_layout()
    
    plt.suptitle("Feature Distributions", fontsize=14, y=1.02)
    plt.show()

# method used to perform pca. In our version, we elected to find how many components explain 95% of the variance. Results in a dataframe
def perform_pca(data_df: pd.DataFrame):

    scaler=RobustScaler()
    trans=PowerTransformer(method='yeo-johnson')
    x,Y=keep_features(data_df=data_df,target='diagnosis',to_drop=['id']) # we get the features and the target

    scaler.fit(x)
    trans.fit(x)
    data_rescaled=scale_data(x,scaler=scaler,transformer=trans) # we scale the data
    print(data_rescaled)

    fraction=0.95 # the fraction of variance that we want to be explained

    # we use sklearn's PCA to perform pca
    pca = PCA(n_components=fraction)
    pca.fit(data_rescaled)
    reduced=pca.transform(data_rescaled)

    print(f"{fraction*100}% of the variance can be explained by {pca.n_components_} components")
    print("The explained variance ratio is: ",(pca.explained_variance_ratio_))

    viz_pca(reduced,Y) # we visualize using just two components, to see whether the class can be seperated

    pca_df=pd.DataFrame(data=reduced)
    pca_df['diagnosis']=Y.values
    
    return pca_df

# method used to scale and transform the data
def scale_data(data,scaler,transformer):
    # we use the RobustScaler as we have outliers and the PowerTransformer to make our data more "normal-like"
    # pipeline = Pipeline([
    # ('scaler', RobustScaler()),  
    # ('transformer', PowerTransformer(method='yeo-johnson'))
    # ])

    pipeline = Pipeline([
    ('scaler', scaler),  
    # ('transformer', transformer)
    ])
    
    # we transform the pipeline
    # data_rescaled=pipeline.fit_transform(data)
    data_rescaled=pipeline.transform(data)

    return data_rescaled

# method used to vizualise two PCA components
def viz_pca(x,y):
    plt.figure(figsize=(10, 8))
    
    scat = plt.scatter(x[:, 0], x[:, 1], c=y.values, cmap='flare', alpha=0.7) # we use scatterplot for the 2 components
    plt.legend(*scat.legend_elements(), title="Labels")
    
    plt.title("Principal Component Analysis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    plt.show()

# method used to count how many times item appears in item_list
def count_apps(item,item_list:list):
    
    count=0
    
    for it in item_list:
        if (it==item):
            count += 1
    
    return count

# method used to count the wins of a method in a summary dataframe
def count_wins(summary:pd.DataFrame):
    # we count how many models we have
    total_models=summary.shape[0]

    dict_app={}

    # for each model, we count how many times it appears in the idxmax list
    for model_num in range(total_models):
        dict_app[model_num]=count_apps(model_num,summary.idxmax())
    
    return dict_app

# we transform the dictionairy we get from count_wins, to dictionairy with {model_name: number_of_wins} entries
def winner_dict(summary:pd.DataFrame):
    
    model_names=summary['Model']
    sum_no_name=summary.drop(columns='Model')

    win_dict=count_wins(summary=sum_no_name)

    for i in range(len(model_names)):
        win_dict[model_names[i]]=win_dict.pop(i)

    return win_dict

# method used to get the winner method, along with all the wins it scored
def get_winner(summary:pd.DataFrame):
    win_dict=winner_dict(summary=summary) # we get the dictionairy with all the wins

    # we sorted in descending order, based on the values it stores
    sorted_win_dict=dict(sorted(win_dict.items(), key=lambda item: item[1],reverse=True))
    winner=list(sorted_win_dict.keys())[0] # the key of the first entry is our winner

    return (winner,sorted_win_dict[winner])

# method used to replace/rename a specific column in a dataframe
def replace_column(df:pd.DataFrame,to_be_replaced,to_be_added):
    data_df=df
    
    # we ensure that the column to be replaced is part of the dataframe and that the new column does not alreadt exist
    if to_be_replaced in data_df.columns and (to_be_added not in data_df.columns):
        data_df.rename(columns={to_be_replaced:to_be_added},inplace=True)
    
    return data_df  

# method used to tune the final winner model that we got through rnCV
def winner_tuning(df:pd.DataFrame,winner):
    # we define our estimators, same as the ones we used in rnCV
    estimators = {
        'LogisticRegression': LogisticRegression,
        'GaussianNB': GaussianNB,
        'LDA': LinearDiscriminantAnalysis,
        'SVM': SVC,
        'RandomForest': RandomForestClassifier,
        'LightGBM': lgb.LGBMClassifier
    }

    # we define the hyperparameter spaces, same as the ones we used in rnCV
    param_spaces = {
        'LogisticRegression': lambda trial: {
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['saga']),
            'C': trial.suggest_float('C', 1e-3, 1e0, log=True),
            'l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1),
            'random_state': trial.suggest_categorical('random_state', [42])
        },
        'GaussianNB': lambda trial: {'var_smoothing': trial.suggest_float('var_smoothing', 1e-2, 1e-1, log=True)},
        'LDA': lambda trial: {'solver':trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
                              'tol':trial.suggest_float('tol', 5*1e-2, 1e-1, log=True)},
        'SVM': lambda trial: {
            'C': trial.suggest_float('C', 5*1e-2, 1e2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'probability': trial.suggest_categorical('probability', [True]),
            'random_state': trial.suggest_categorical('random_state', [42])            
        },
        'RandomForest': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state': trial.suggest_categorical('random_state', [42])            
        },
        'LightGBM': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'random_state': trial.suggest_categorical('random_state', [42]),
            'verbosity': trial.suggest_categorical('verbosity', [-1])
        }
    }

    # we initialize a rnCV class instance
    rncv=rnCV(data_df=df, estimators=estimators,params=param_spaces, r=10, n=5, k=3, random_state=42)
    winner_estim=estimators[winner] # we get the actual estimator, not just his name
    winner_params=rncv.tune_winner(winner=winner) # and we use rnCV tune_winner to tune it

    print(f'For model {winner} the best parameters are {winner_params}') # we print the best parameters for clarity

    return winner_estim(**winner_params)


# method used to print the confidence interval of bootstrapping as actual intervals of a single metric
def metric_ci(y_true, y_pred, metric, is_proba=False, proba=None, n_samples=5000, seed=42):
    
    rng=np.random.RandomState(seed)
    stats=[]
    
    # we perform bootstrapping
    for sample in range(n_samples):
        
        idx=rng.choice(len(y_true), len(y_true), replace=True) # we choose a random index
        
        y_sample=y_true[idx] if isinstance(y_true, np.ndarray) else y_true.iloc[idx] # we check whether it's and np ndarray or a dataframe/series
        pred_sample=proba[idx] if is_proba else y_pred[idx] # we get the proper prediction type based on the metric
        
        stats.append(metric(y_sample, pred_sample)) # apply the metric given
    
    return np.percentile(stats, [2.5, 97.5]) # we return the two ends of the interval


# method used to print the confidence interval of bootstrapping as actual intervals([start,end]) of all metrics
def bootstrap_model_intervals(df_dev:pd.DataFrame,df_val:pd.DataFrame, model):
    # our metrices
    metrics = {
        'Balanced Accuracy': (balanced_accuracy_score, False),
        'F1 Score': (f1_score, False),
        'Precision': (precision_score, False),
        'Recall': (recall_score, False),
        'MCC': (matthews_corrcoef, False),
        'ROC AUC': (roc_auc_score, True),
        'PR AUC': (average_precision_score, True)
    }


    # we get the x and y of the dev set
    x_dev, y_dev=keep_features(data_df=df_dev,target='diagnosis',to_drop=['id'])
    y_dev=encode(y_dev)
    
    scaler=RobustScaler()
    trans=PowerTransformer(method='yeo-johnson')
    imp=IterativeImputer(random_state=42)

    scaler.fit(x_dev)
    trans.fit(x_dev)
    imp.fit(x_dev)
    # we scale and we impute here as to avoid data leakage
    x_dev=impute(x_dev,imp=imp)
    x_dev=scale_data(x_dev,scaler=scaler,transformer=trans)

    # we get the x and y of the val set    
    x_val, y_val=keep_features(data_df=df_val,target='diagnosis',to_drop=['id'])
    y_val=encode(y_val)

    # we scale and we impute here as to avoid data leakage
    x_val=impute(x_val,imp=imp)
    x_val=scale_data(x_val,scaler=scaler,transformer=trans)

    # we fit the model on dev
    model.fit(x_dev,y_dev)

    # predict on evaluation set
    y_pred=model.predict(x_val)
    y_proba=model.predict_proba(x_val)[:, 1]


    print("Bootstrapped 95% CIs (Model trained on dev, tested on val):")
    
    # we iterate over all the metric methods
    for name, (fn, is_proba) in metrics.items():
        # we calculate the interval of each specific metric using the actual function (fn) of the metric
        ci=metric_ci(
            y_val,
            y_pred if not is_proba else y_proba,
            fn,
            is_proba=is_proba,
            proba=y_proba if is_proba else None
        )
        # we print the method and the interval
        print(f"{name:15s}: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # specificity and NPV as point estimates (not bootstrapped here)
    tn, fp, fn_, tp=confusion_matrix(y_val, y_pred).ravel()
    
    specificity=tn / (tn + fp) if (tn + fp) > 0 else 0
    npv=tn / (tn + fn_) if (tn + fn_) > 0 else 0
    
    print(f"Specificity         : {specificity:.4f}")
    print(f"NPV                 : {npv:.4f}")


# method used to calculate the metrics score of all methods need for the violinplot
def metric_ci_plot(y_true, y_pred, proba, n_samples=1000, seed=42):
    rng = np.random.RandomState(seed)
    # the scores of each metric
    metrics = {
        'Balanced Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': [],
        'MCC': [],
        'ROC AUC': [],
        'PR AUC': []
    }

    for sample in range(n_samples):
        
        idx=rng.choice(len(y_true), len(y_true), replace=True) # we choose a random index
        
        y_bs=y_true[idx] if isinstance(y_true, np.ndarray) else y_true.iloc[idx] # we check whether we have a ndarray or a Dataframe/Series
        y_pred_bs=y_pred[idx] # actual class prediction
        y_proba_bs=proba[idx] # class probabilities prediction

        # we calculate each metric by using either the actual class or the class probabilites prediction
        metrics['Balanced Accuracy'].append(balanced_accuracy_score(y_bs, y_pred_bs))
        metrics['F1 Score'].append(f1_score(y_bs, y_pred_bs))
        metrics['Precision'].append(precision_score(y_bs, y_pred_bs))
        metrics['Recall'].append(recall_score(y_bs, y_pred_bs))
        metrics['MCC'].append(matthews_corrcoef(y_bs, y_pred_bs))
        metrics['ROC AUC'].append(roc_auc_score(y_bs, y_proba_bs))
        metrics['PR AUC'].append(average_precision_score(y_bs, y_proba_bs))

    return metrics

# method used to plot the confidence intervals of bootstrapping. It used violin plots
def bootstrap_model_plot(df_dev:pd.DataFrame,df_val:pd.DataFrame, model):

    # we get the x and y of the dev set
    x_dev, y_dev=keep_features(data_df=df_dev,target='diagnosis',to_drop=['id'])
    y_dev=encode(y_dev)

    scaler=RobustScaler()
    trans=PowerTransformer(method='yeo-johnson')
    imp=IterativeImputer(random_state=42)

    scaler.fit(x_dev)
    trans.fit(x_dev)
    imp.fit(x_dev)

    # we impute and we scale here as to avoid data leakage
    x_dev=impute(x_dev,imp=imp)
    x_dev=scale_data(x_dev,scaler=scaler,transformer=trans)

    # we get the x and y of the val set    
    x_val, y_val=keep_features(data_df=df_val,target='diagnosis',to_drop=['id'])
    y_val=encode(y_val)

    # we impute and we scale here as to avoid data leakage
    x_val=impute(x_val,imp=imp)
    x_val=scale_data(x_val,scaler=scaler,transformer=trans)

    # we fit the model on dev dataset
    model.fit(x_dev,y_dev)

    # we predict on evaluation set
    y_pred=model.predict(x_val) # actual classes prediction
    y_proba=model.predict_proba(x_val)[:, 1] # classe probabilities prediction

    # we run bootstrapping
    bootstrap_scores=metric_ci_plot(y_val, y_pred, y_proba, n_samples=5000)

    # we create a dataframe with the results to visualize with violin plots
    df_bootstrap = pd.DataFrame({
        'Metric': sum([[metric]*len(scores) for metric, scores in bootstrap_scores.items()], []),
        'Score': sum(bootstrap_scores.values(), [])
    })

    # we plot using violin plots
    plt.figure(figsize=(10, 6))
    
    sns.violinplot(data=df_bootstrap, x='Metric', y='Score', inner='quartile', palette='pastel')
    
    plt.title("Bootstrapped 95% CI Distributions (val set)")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# method used to perform bootstrapping. It prints intervals and visualizes them with violin plots
def bootstrap_model(df_dev:pd.DataFrame,df_val:pd.DataFrame, model):
    bootstrap_model_intervals(df_dev=df_dev,df_val=df_val,model=model)
    bootstrap_model_plot(df_dev=df_dev,df_val=df_val,model=model)

# method used to save the winner model. It actually saves a pipeline
def save_winner(path,winner,winner_name):
    # the directory where the pipeline will be saved
    models_dir="../models"
    model_io=IO(models_dir) # class used to handle the actual saving

    df=pd.read_csv(path)

    # we get the x and y of the given dataset
    x_full,y_full=keep_features(data_df=df)
    y_full=encode(y_full)

    # we drop the target from the features
    x_full=df.drop(columns="diagnosis")

    # we save the features we used so that the pipeline may use them during inference
    feature_columns=x_full.columns.tolist()
    joblib.dump(feature_columns, "../models/feature_columns.pkl")

    # the pipeline that handles the scaling and transformation
    scale_pipeline=Pipeline([
    ('scaler', RobustScaler()),  
    ('transformer', PowerTransformer(method='yeo-johnson'))
    ])

    # the pipeline that handles the Imputing
    preprocessor=Pipeline([
        ('imputer', IterativeImputer(random_state=42)),  
        ('scaler', scale_pipeline)                    
    ])

    # the complete pipeline that will be saved. It includes column selection, preprocessing and the model
    winner_pipeline=Pipeline([
        ('select_columns',ColumnSelector(columns=feature_columns)),
        ('preproccesing',preprocessor),
        ('model',winner)
    ])

    # we fit the model on the dataset
    winner_pipeline.fit(X=x_full,y=y_full)
    
    print(f"Saving winner model ({winner_name}) with name winner.pkl")

    # we save the model
    model_io.save(model=winner_pipeline,name='winner')


# method used to perform inference on the dataset that resides in test_df_path
def infere_with_winner(test_df_path):

    # we load the dataset
    test_df=pd.read_csv(test_df_path)

    # the directory where the winner lies
    models_dir="../models"
    model_io=IO(models_dir) # this class will handle the loading

    winner=model_io.load(name='winner') # we load the model

    preds=winner.predict(test_df) # we predict
    
    # we save the predictions in a dataframe
    test_df["predicted_diagnosis"] = preds
    print(test_df[["predicted_diagnosis"]].head())

    test_df.to_csv('../data/predictions.csv',index=False)

# method used to get the best top coefficients of Logistic Regression. Returns a dataframe
def get_top_coefficients(top,names):

    # the directory where the winner lies
    models_dir="../models"
    model_io=IO(models_dir) # this class will handle the loading

    winner=model_io.load(name='winner') # we load the model

    cf=(winner.named_steps['model'].coef_).flatten() # the coefficients of the model
    features=names # the name of the features

    # the dataframe to be returned, sorted in descending absolute coefficient value
    cf_df=pd.DataFrame({
        'Feature':features,
        'Coefficient':cf,
        'abs_coeff': np.abs(cf)
    }).sort_values(by='abs_coeff',ascending=False)

    print(cf_df.head(top))

    return cf_df

# method used to plot the best top coefficients of LogisticRegression. Needs a dataframe as input
def plot_best_coefficients(cf_df:pd.DataFrame,top):
    plt.figure(figsize=(10,6))

    sns.barplot(data=cf_df.head(top),y='Feature',x='Coefficient',palette='magma')
    
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'Top {top} most influential features (LogisticRegression Coefficients)')
    
    plt.tight_layout()
    plt.show()

# method used to get the most top influentiel coefficients of a Logistic Regression model and plot them
def top_coefficients_winner(top,names):
    cf_df=get_top_coefficients(top=top,names=names)
    plot_best_coefficients(cf_df=cf_df,top=top)

    return cf_df