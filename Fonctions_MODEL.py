from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import BinaryEncoder


def encode_cat_col(
    df: pd.DataFrame,
    col_name: str,
    encoding_type: str,
) -> tuple[pd.DataFrame, object]:
    """
    Encode une seule variable catégorielle selon la méthode choisie.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame source.
    col_name : str
        Nom de la colonne à encoder.
    encoding_type : str
        Type d'encodage à appliquer.
        Valeurs possibles :
        - "onehot"
        - "binary"
        - "ordinal"

    Retours
    -------
    tuple[pd.DataFrame, object]
        - Le DataFrame avec la colonne encodée
        - L'encodeur entraîné

    Notes
    -----
    - `onehot` convient aux variables nominales à faible cardinalité.
    - `binary` convient aux variables nominales à forte cardinalité.
    - `ordinal` convient aux variables ordinales.
    - Pour `binary`, il faut installer `category-encoders`.
    """
    if col_name not in df.columns:
        raise ValueError(f"La colonne '{col_name}' n'existe pas dans le DataFrame.")

    df_encoded = df.copy()

    if encoding_type == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        encoded_array = encoder.fit_transform(df_encoded[[col_name]])
        encoded_cols = encoder.get_feature_names_out([col_name])

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoded_cols,
            index=df_encoded.index,
        )

        df_encoded = pd.concat(
            [df_encoded.drop(columns=[col_name]), encoded_df],
            axis=1,
        )

        return df_encoded, encoder

    if encoding_type == "binary":
        encoder = BinaryEncoder(cols=[col_name])

        encoded_df = encoder.fit_transform(df_encoded[[col_name]])

        df_encoded = pd.concat(
            [df_encoded.drop(columns=[col_name]), encoded_df],
            axis=1,
        )

        return df_encoded, encoder

    if encoding_type == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        encoded_array = encoder.fit_transform(df_encoded[[col_name]])
        df_encoded[col_name] = encoded_array.astype(int)

        return df_encoded, encoder

    raise ValueError(
        "encoding_type doit être parmi : 'onehot', 'binary', 'ordinal'."
    )


def evaluate_regression_model(model, X, y, test_size=0.2):
    """
    Évalue un modèle de régression à l'aide d'un unique découpage train/test.

    La fonction sépare les données en un jeu d'entraînement et un jeu de test,
    entraîne le modèle sur le jeu d'entraînement, puis calcule plusieurs
    métriques de régression sur les deux sous-ensembles.

    Paramètres
    ----------
    model : estimator object
        Modèle de régression implémentant les méthodes `fit(X, y)` et `predict(X)`.
    X : pd.DataFrame ou array-like
        Matrice des variables explicatives.
    y : pd.Series ou array-like
        Vecteur cible.
    test_size : float, default=0.2
        Proportion des données réservée au jeu de test.

    Retours
    -------
    dict
        Dictionnaire contenant les métriques d'entraînement et de test :
        - "Train R2"
        - "Test R2"
        - "Train MAPE (%)"
        - "Test MAPE (%)"
        - "Train MAE"
        - "Test MAE"
        - "Train RMSE"
        - "Test RMSE"

    Notes
    -----
    - Le découpage est reproductible grâce à `random_state=42`.
    - Le `MAPE` est renvoyé en pourcentage.
    - Le `MAPE` peut devenir instable si la cible contient des valeurs proches de zéro.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),

        "Train MAPE (%)": mean_absolute_percentage_error(y_train, y_train_pred) * 100,
        "Test MAPE (%)": mean_absolute_percentage_error(y_test, y_test_pred) * 100,

        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),

        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }

    return metrics


import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
)


def _safe_inverse(arr, inverse_func, clip_min=None, clip_max=None):
    a = np.asarray(arr, dtype=np.float64)
    if clip_min is not None or clip_max is not None:
        a = np.clip(a, clip_min, clip_max)
    return inverse_func(a)

def evaluate_regression_model_cv(model, X, y, cv, inverse_target_func=None):
    """
    Évalue un modèle de régression via validation croisée K-Fold.

    La fonction entraîne le modèle sur chaque fold, prédit sur les sous-ensembles
    d'entraînement et de validation, puis calcule les métriques moyennes.

    Elle gère aussi le cas d'une cible transformée (ex. `y = log1p(target)`) :
    si `inverse_target_func` est fourni (ex. `np.expm1`), les métriques sont
    calculées après retour à l'échelle réelle.

    Paramètres
    ----------
    model : estimator object
        Modèle de régression implémentant `fit(X, y)` et `predict(X)`.
    X : pd.DataFrame
        Variables explicatives. Utilise `.iloc` pour l'indexation des folds.
    y : pd.Series ou np.ndarray
        Cible. Utilise `.iloc` si disponible.
    cv : int
        Nombre de folds pour la validation croisée.
    inverse_target_func : callable, optionnel
        Fonction inverse de la transformation appliquée à la cible.
        Exemple : `np.expm1` si la cible a été transformée avec `np.log1p`.
        Si `None`, les métriques sont calculées dans l'échelle actuelle de `y`.

    Retours
    -------
    dict
        Dictionnaire des métriques moyennes train/test :
        - "Train R2", "Test R2"
        - "Train MAPE (%)", "Test MAPE (%)"
        - "Train MAE", "Test MAE"
        - "Train RMSE", "Test RMSE"

    Notes
    -----
    - K-Fold est mélangé avec `random_state=42` pour la reproductibilité.
    - Le MAPE est renvoyé en pourcentage.
    - Si la cible contient des valeurs proches de 0, le MAPE peut être instable.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    r2_train, r2_test = [], []
    mape_train, mape_test = [], []
    mae_train, mae_test = [], []
    rmse_train, rmse_test = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if inverse_target_func is not None:
            # Bornes apprises sur y_train (échelle transformée), marge optionnelle
            y_min = float(np.min(y_train))
            y_max = float(np.max(y_train))
            margin = 0.5
            clip_min = y_min - margin
            clip_max = y_max + margin

            y_train_eval = _safe_inverse(y_train, inverse_target_func, clip_min, clip_max)
            y_test_eval = _safe_inverse(y_test, inverse_target_func, clip_min, clip_max)
            y_train_pred_eval = _safe_inverse(y_train_pred, inverse_target_func, clip_min, clip_max)
            y_test_pred_eval = _safe_inverse(y_test_pred, inverse_target_func, clip_min, clip_max)

        else:
            y_train_eval = y_train
            y_test_eval = y_test
            y_train_pred_eval = y_train_pred
            y_test_pred_eval = y_test_pred

        r2_train.append(r2_score(y_train_eval, y_train_pred_eval))
        r2_test.append(r2_score(y_test_eval, y_test_pred_eval))

        mape_train.append(mean_absolute_percentage_error(y_train_eval, y_train_pred_eval) * 100)
        mape_test.append(mean_absolute_percentage_error(y_test_eval, y_test_pred_eval) * 100)

        mae_train.append(mean_absolute_error(y_train_eval, y_train_pred_eval))
        mae_test.append(mean_absolute_error(y_test_eval, y_test_pred_eval))

        rmse_train.append(np.sqrt(mean_squared_error(y_train_eval, y_train_pred_eval)))
        rmse_test.append(np.sqrt(mean_squared_error(y_test_eval, y_test_pred_eval)))

    return {
        "Train R2": np.mean(r2_train),
        "Test R2": np.mean(r2_test),
        "Train MAPE (%)": np.mean(mape_train),
        "Test MAPE (%)": np.mean(mape_test),
        "Train MAE": np.mean(mae_train),
        "Test MAE": np.mean(mae_test),
        "Train RMSE": np.mean(rmse_train),
        "Test RMSE": np.mean(rmse_test),
    }


def print_cv_results(models, X, y, cv=5, inverse_target_func=None):
    """
    Affiche les performances CV de plusieurs modèles de régression.

    Pour chaque modèle du dictionnaire, la fonction appelle
    `evaluate_regression_model_cv` puis affiche les métriques train/test
    de façon lisible.

    Paramètres
    ----------
    models : dict[str, estimator object]
        Dictionnaire `{nom_modele: estimateur}`.
    X : pd.DataFrame
        Variables explicatives.
    y : pd.Series ou np.ndarray
        Cible.
    cv : int, default=5
        Nombre de folds K-Fold.
    inverse_target_func : callable, optionnel
        Fonction inverse de transformation de la cible pour afficher des
        métriques dans l'échelle réelle (ex. `np.expm1` pour une cible `log1p`).

    Retours
    -------
    None
        La fonction affiche les résultats dans la console.
    """
    for name, model in models.items():
        metrics = evaluate_regression_model_cv(
            model=model,
            X=X,
            y=y,
            cv=cv,
            inverse_target_func=inverse_target_func,
        )

        print(f"\n{name}")
        print(
            f"Train | R²: {metrics['Train R2']:.4f} | "
            f"MAPE: {metrics['Train MAPE (%)']:.2f}% | "
            f"MAE: {metrics['Train MAE']:.4f} | "
            f"RMSE: {metrics['Train RMSE']:.4f}"
        )
        print(
            f"Test  | R²: {metrics['Test R2']:.4f} | "
            f"MAPE: {metrics['Test MAPE (%)']:.2f}% | "
            f"MAE: {metrics['Test MAE']:.4f} | "
            f"RMSE: {metrics['Test RMSE']:.4f}"
        )


