import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from training.logging_config import get_logger

logger = get_logger(log_file="logs/model_training.log")


def train_model(X_train, y_train, X_test, y_test):

    logger.info("Model training started")

    os.makedirs("outputs/plots", exist_ok=True)

    y_train_log = np.log1p(y_train)

    # Train Multiple Models

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train_log)
        y_pred = np.expm1(model.predict(X_test))

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse})
        logger.info(f"{name}: R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/model_comparison.csv", index=False)
    logger.info("Model comparison saved")

    best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
    logger.info(f"Best model: {best_model_name}")


    # Hyperparameter Tuning (GB)

    gb_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_params,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )

    gb_grid.fit(X_train, y_train_log)

    logger.info(f"Best GB params: {gb_grid.best_params_}")
    logger.info(f"Best CV R²: {gb_grid.best_score_:.4f}")

    best_model = gb_grid.best_estimator_

    #  Final Evaluation

    train_pred = np.expm1(best_model.predict(X_train))
    test_pred = np.expm1(best_model.predict(X_test))

    logger.info(f"Train R²: {r2_score(y_train, train_pred):.4f}")
    logger.info(f"Test R²: {r2_score(y_test, test_pred):.4f}")

    # Regularization Plot

    alphas = [0.01, 0.1, 1, 10, 100]
    ridge_scores = []
    lasso_scores = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train_log)
        ridge_scores.append(
            r2_score(y_test, np.expm1(ridge.predict(X_test)))
        )

        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(X_train, y_train_log)
        lasso_scores.append(
            r2_score(y_test, np.expm1(lasso.predict(X_test)))
        )

    plt.figure(figsize=(10, 5))
    plt.plot(alphas, ridge_scores, marker='o', label='Ridge')
    plt.plot(alphas, lasso_scores, marker='s', label='Lasso')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('R²')
    plt.title('Regularization Effect')
    plt.legend()
    plt.grid(True)

    plot_path = "outputs/plots/regularization_effect.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Regularization plot saved: {plot_path}")
    logger.info("Model training completed successfully")

    return best_model
