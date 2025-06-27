from enum import Enum
from sklearn.feature_selection import SelectKBest, f_classif


class FeatureSelectionMethod(Enum):
    ANOVA = "ANOVA"


def select_features(method: FeatureSelectionMethod, X_train, y_train, n_features: int):
    match method:
        case FeatureSelectionMethod.ANOVA:
            selector = SelectKBest(f_classif, k=n_features)
            selector = selector.fit(X_train, y_train)
            selected_features = selector.get_support()
            return selected_features
        case _:
            raise NotImplementedError(f"The feature selection method '{method.value}' is not implemented yet.")
