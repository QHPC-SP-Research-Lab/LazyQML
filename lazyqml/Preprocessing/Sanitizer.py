# Importing from
from lazyqml.Interfaces.iPreprocessing import Preprocessing

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing   import MinMaxScaler, RobustScaler

import numpy as np

class Sanitizer(Preprocessing):
    def __init__(self, imputerCat, imputerNum):
        # scalers = [("robust", RobustScaler()),
        #            ("scaler", MinMaxScaler(feature_range=(0, 2*np.pi), clip=True))]
        
        scalers = [("scaler", StandardScaler())]
        
        cat_steps = [("imputer", imputerCat)] + scalers
        num_steps = [("imputer", imputerNum)] + scalers

        self.categorical_transformer = Pipeline(steps=cat_steps)
        self.numeric_transformer = Pipeline(steps=num_steps)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", self.numeric_transformer, selector(dtype_exclude="category")),
                ("categorical_low", self.categorical_transformer, selector(dtype_include="category")),
            ]
        )

    def fit(self, X):
        return self.preprocessor.fit(X)

    def fit_transform(self, X):
        return self.preprocessor.fit_transform(X)

    def transform(self, X):
        return self.preprocessor.transform(X)