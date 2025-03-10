from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def regression_model(name: str, numeric_cols, cat_cols, binary_cols, **kwargs):

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough')

    if name == "RandomForest":
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Preprocessing step
            ("regressor",  clone(RandomForestRegressor(n_estimators=500, **kwargs))),])
        return pipeline
    elif name == "GradientBoosting":
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Preprocessing step
            ("regressor",  clone(GradientBoostingRegressor(**kwargs))),])
        return pipeline
    elif name == "Lasso":
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Preprocessing step(
            ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False),),
            ("regressor", clone(Lasso(**kwargs))),])
        return pipeline
    else:
        raise ValueError(f"Model {name} not recognized")


def classification_model(name: str, numeric_cols, cat_cols, binary_cols, **kwargs):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough')

    if name == "RandomForest":
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Preprocessing step
            ("classifier",  clone(RandomForestClassifier(n_estimators=500, **kwargs))),])
        return pipeline
    elif name == "GradientBoosting":
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Preprocessing step
            ("classifier",  clone(GradientBoostingClassifier(**kwargs))),])
        return pipeline
    elif name == "Lasso":
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False),),
            ("classifier", clone(LogisticRegression(penalty="l1", solver="liblinear", **kwargs)),),])
        return pipeline
    else:
        raise ValueError(f"Model {name} not recognized")


def hyperparameters_grid(name: str, classification: bool = False):
    if name == "RandomForest":
        if classification:
            return {"classifier__max_depth": [1, 2, 3, 5, 10, 20],
                    "classifier__min_samples_leaf": [5, 10, 15, 20, 30, 50],}
        else:
            return {"regressor__max_depth": [1, 2, 3, 5, 10, 20],
                    "regressor__min_samples_leaf": [5, 10, 15, 20, 30, 50],}

    elif name == "GradientBoosting":
        if classification:
            return {"classifier__n_estimators": [5, 10, 25, 50, 100, 200, 500],
                    "classifier__learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    "classifier__max_depth": [1, 2, 3, 5, 10],}
        else:
            return {"regressor__n_estimators": [5, 10, 25, 50, 100, 200, 500],
                    "regressor__learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    "regressor__max_depth": [1, 2, 3, 5, 10],}

    elif name == "Lasso":
        if classification:
            return {"classifier__C": [0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 1]}
        else:
            return {"regressor__alpha": [0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 1]}
    else:
        raise ValueError(f"Model {name} not recognized")
