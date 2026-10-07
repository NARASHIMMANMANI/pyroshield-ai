import joblib

from sklearn.ensemble import StackingClassifier

from sklearn.linear_model import LogisticRegression


class StackingModel:

    def __init__(self,
                 rf,
                 xgb,
                 cat,
                 ft_model,
                 kan_model):

        estimators = [

            ("RandomForest", rf),
            ("XGBoost", xgb),
            ("CatBoost", cat)

        ]

        self.model = StackingClassifier(

            estimators=estimators,

            final_estimator=LogisticRegression(),

            stack_method="predict_proba",

            passthrough=False,

            cv=5,

            n_jobs=-1

        )

        self.ft_model = ft_model
        self.kan_model = kan_model

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)

    def save(self, path):

        joblib.dump(self.model, path)