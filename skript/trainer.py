import pandas as pd 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor 


def data_reader(path, filename): 
    df=pd.read_csv(path+filename, sep=';', thousands=';')
    features=df.drop('nachforderung', axis=1)
    target=df.loc[:, 'nachforderung']
    return features, target


#for retraining 
class Trainer():  
    """Class for Model Training. 
    
    Attrs:
        features (Pandas DataFrame): The features for Training
        target (Pandas Series): The Target for Training

    Methods: 
        fit(quantile_low, quantile_high): Fits a random forest regressor 
        outlier_removal(): removes outliers, used in fit()  

    """
    def __init__(self, features_train, target_train):   
        self.features=features_train
        self.target=target_train

    def fit(self, quantile_low=0.1, quantile_high=0.9): 
        """Fit a random forest regressor and return fitted model.
        
        Attrs:
            features (Pandas DataFrame): The features for Training
            target (Pandas Series): The Target for Training

        Return: fitted model

        """
        self.outlier_removal(quantile_low, quantile_high)
        rf=RandomForestRegressor(random_state=0, max_depth=15) #here hard coded, could be made more flexible with search spaces
        rf.fit(self.features, self.target)
        print('random forest fitted')
        return rf 

    def outlier_removal(self, quantile_low=0.1, quantile_high=0.9):
        """remove outlier which are outside of 1.5*IQR, where IQR is defined as the range corresponding to quantile_high minus quantile_low. Performed on each column in dataframe. 
        Return dataframe with outliers removed"""
        df_new=self.features.copy()
        for col in df_new.columns:
            iqr=df_new.loc[:,'feature_1'].quantile(q=quantile_high)-df_new.loc[:,'feature_1'].quantile(q=quantile_low)
            upper=df_new.loc[:,'feature_1'].quantile(q=quantile_high)+1.5*iqr
            lower=df_new.loc[:,'feature_1'].quantile(q=quantile_low)-1.5*iqr
            mask=(df_new.loc[:,'feature_1']>lower) & (df_new.loc[:,'feature_1']<upper)
            df_new=df_new.loc[mask, :]
        diff=len(self.features)-len(df_new)
        self.features=df_new
        print(f'{diff} lines removed')
        merged=pd.merge(self.target, self.features, left_index=True, right_index=True)
        self.target=merged.loc[:, 'nachforderung']
        return 