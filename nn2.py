import os
import logging
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.regularizers import l2
from category_encoders import OneHotEncoder
import tensorflow.keras.backend as K
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


class ConfigLoader:
    @staticmethod
    def load_config(config_path='config.json'):
        with open(config_path, 'r') as config_file:
            return json.load(config_file)


class DataLoader:
    @staticmethod
    def load_data(data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found. Ensure it is in the correct directory.")
        return pd.read_csv(data_path)


class DataPreprocessor:
    @staticmethod
    def remove_outliers(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    @staticmethod
    def feature_engineering(data):
        data = data.loc[(data['Area_sqm'] < 1000) & (data['Price_USD'] < 2400000)].copy()
        logger.info(f"Data size after applying limit: {data.shape}")

        # Логарифм ціни об'єкта нерухомості
        data['Log_Price'] = np.log1p(data['Price_USD'])

        # Логарифм відстані до центру міста
        data['Log_DistanceToCenter_km'] = np.log1p(data['DistanceToCenter_km'])

        # Логарифм відстані до найближчого метро
        data['Log_DistanceToNearestMetro_km'] = np.log1p(data['DistanceToNearestMetro_km'])

        # Середня ціна в межах району
        data['Average_Price_Per_District'] = data.groupby('District')['Price_USD'].transform('mean')

        # Категорія розміру кімнат
        data['Room_Class'] = pd.cut(data['Rooms'], bins=[0, 2, 4, float('inf')], labels=['small', 'medium', 'large'])

        # Площа на одну кімнату
        data['Area_per_room'] = data['Area_sqm'] / data['Rooms']

        # Логарифм площі на кімнату
        data['Log_Area_per_room'] = np.log1p(data['Area_per_room'])

        # Взаємодія між кількістю кімнат і площею
        data['Rooms_Area_Interaction'] = data['Rooms'] * data['Area_sqm']

        # Взаємодія між ціною і відстанню до центру
        data['Price_Distance_Interaction'] = data['Price_USD'] / (data['DistanceToCenter_km'] + 1)

        # Взаємодія між відстанню до центру та до метро
        data['Distance_Interaction'] = data['DistanceToCenter_km'] * data['DistanceToNearestMetro_km']

        # Взаємодія між широтою та довготою
        data['Lat_Long_Interaction'] = data['Latitude'] * data['Longitude']

        # Вагова середня ціна
        data['Weighted_Avg_Price'] = data.groupby('Neighborhood').apply(
            lambda x: np.average(x['Price_USD'], weights=x['Area_sqm'] * x['Rooms'])
        ).reindex(data['Neighborhood']).values

        data['Metro_Accessibility'] = pd.cut(
            data['DistanceToNearestMetro_km'],
            bins=[-float('inf'), 0.5, 1, 2, 5, float('inf')],
            labels=['Very Close', 'Close', 'Moderate', 'Far', 'Very Far']
        )

        data['BuildingType'] = np.where(
            (data['BuildingType'] == 'unknown') & ((data['Area_sqm'] > 70) | (data['Rooms'] >= 3)),
            'монолітно-каркасний',
            data['BuildingType']
        )

        data.drop(columns=['DistanceToCenter_km', 'DistanceToNearestMetro_km', 'Area_per_room', 'Weighted_Avg_Price',
                           'Area_sqm', 'Rooms', 'Latitude', 'Longitude'],
                  inplace=True)
        return data

    @staticmethod
    def prepare_data(data):
        columns_to_drop = ['PricePerSqm_USD', 'Description', 'DateCreated', 'ID']
        X = data.drop(columns=['Price_USD', 'Log_Price'] + columns_to_drop)
        y = data['Log_Price']
        return X, y


class ModelBuilder:
    @staticmethod
    def build_model(input_dim, layers, activation, dropout_rates, use_batch_norm, use_dropout, learning_rate, l2_reg,
                    resnet_blocks):
        inputs = Input(shape=(input_dim,))
        x = inputs
        for i, units in enumerate(layers):
            for _ in range(resnet_blocks):
                prev_x = x
                x = Dense(units, kernel_regularizer=l2(l2_reg))(x)
                if use_batch_norm:
                    x = BatchNormalization()(x)
                x = Activation(activation)(x)
                if use_dropout and dropout_rates[i] > 0:
                    x = Dropout(dropout_rates[i])(x)
                if x.shape[-1] != prev_x.shape[-1]:
                    prev_x = Dense(x.shape[-1], activation=activation, kernel_regularizer=l2(l2_reg))(prev_x)
                x = tf.keras.layers.add([x, prev_x])
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=LogCosh(), metrics=['mae', 'mse', r2_keras])
        return model

    @staticmethod
    def get_callbacks(fold):
        checkpoint_path = f'best_model_fold_{fold}.keras'
        return [
            EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-4, restore_best_weights=True, verbose=1),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        ]

    @staticmethod
    def train_model(model, X_train, y_train, X_val, y_val, config, fold):
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=ModelBuilder.get_callbacks(fold),
            verbose=1
        )
        return history.history


class ModelEvaluator:
    @staticmethod
    def evaluate_model(model_path, X_val, y_val):
        best_model = load_model(model_path, custom_objects={"LogCosh": LogCosh, "r2_keras": r2_keras})
        y_pred = best_model.predict(X_val).flatten()
        y_pred = np.expm1(y_pred)
        y_val_exp = np.expm1(y_val)

        # Розрахунок метрик
        mse = mean_squared_error(y_val_exp, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_exp, y_pred)
        r2 = r2_score(y_val_exp, y_pred)

        return rmse, mae, r2, y_val_exp, y_pred


class Visualization:
    @staticmethod
    def save_figure(fig, filename, format='html'):
        os.makedirs('image', exist_ok=True)
        if format == 'png':
            fig.write_image(f'image/{filename}.png')
        elif format == 'html':
            fig.write_html(f'image/{filename}.html')

    @staticmethod
    def plot_training_history(history_data, fold):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=history_data['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='royalblue')
        ))
        fig.add_trace(go.Scatter(
            y=history_data['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='firebrick')
        ))

        fig.update_layout(
            title=f'Training History for Fold {fold}',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_dark'
        )
        Visualization.save_figure(fig, f'training_history_fold_{fold}')
        return fig

    @staticmethod
    def plot_predictions(y_test, y_pred, fold):
        fig = px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': 'Actual Prices', 'y': 'Predicted Prices'},
            title=f'Predictions vs Actuals for Fold {fold}',
            color_discrete_sequence=['mediumseagreen']
        )
        fig.add_shape(
            type='line',
            line=dict(dash='dash', color='red'),
            x0=y_test.min(),
            y0=y_test.min(),
            x1=y_test.max(),
            y1=y_test.max()
        )
        fig.update_layout(
            template='plotly_dark'
        )
        Visualization.save_figure(fig, f'predictions_fold_{fold}')
        return fig

    @staticmethod
    def plot_residuals(y_test, y_pred, fold):
        residuals = y_test - y_pred
        fig = px.histogram(
            residuals,
            nbins=50,
            labels={'value': 'Residuals'},
            title=f'Residual Distribution for Fold {fold}'
        )
        fig.update_layout(
            xaxis_title='Residuals (Actual - Predicted)',
            yaxis_title='Frequency'
        )
        Visualization.save_figure(fig, f'residuals_fold_{fold}')
        return fig

    @staticmethod
    def plot_feature_importance(model, feature_names, fold, use_shap=False, X_train=None):
        if use_shap and X_train is not None:
            # Використання SHAP для важливості ознак
            explainer = shap.DeepExplainer(model, X_train[:100])
            shap_values = explainer.shap_values(X_train[:100])

            # Візуалізація SHAP Summary Plot
            shap.summary_plot(shap_values, X_train[:100], feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot for Fold {fold}')
            plt.savefig(f'image/shap_summary_fold_{fold}.png')
            plt.close()
        else:
            # Альтернативний метод: аналіз ваг першого Dense шару
            first_layer_weights = model.layers[1].get_weights()[0]
            feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                color='Importance',
                color_continuous_scale='Viridis',
                orientation='h',
                title=f'Feature Importance for Fold {fold}'
            )
            fig.update_layout(
                xaxis_title='Mean Absolute Weight',
                yaxis_title='Feature',
                coloraxis_colorbar=dict(title="Importance")
            )
            Visualization.save_figure(fig, f'feature_importance_fold_{fold}')
            return fig

    @staticmethod
    def plot_correlation_matrix(data_encoded):
        correlation_matrix = data_encoded.corr()
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='Viridis',
            title='Correlation Matrix of Features'
        )
        Visualization.save_figure(fig, 'correlation_matrix')
        return fig

    @staticmethod
    def plot_feature_distribution(X_train, X_val, feature_name, fold):
        plt.figure(figsize=(10, 5))
        sns.histplot(X_train[:, feature_name], color='blue', label='Train', kde=True, stat="density", linewidth=0,
                     alpha=0.5)
        sns.histplot(X_val[:, feature_name], color='red', label='Validation', kde=True, stat="density", linewidth=0,
                     alpha=0.5)
        plt.legend()
        plt.title(f'Distribution of {feature_name} in Train and Validation Sets for Fold {fold}')
        plt.savefig(f'image/feature_distribution_{feature_name}_fold_{fold}.png')
        plt.close()


# Main function
def main():
    config = ConfigLoader.load_config()

    data = DataLoader.load_data('dataset.csv')

    data = DataPreprocessor.feature_engineering(data)

    X, y = DataPreprocessor.prepare_data(data)

    kf = KFold(n_splits=config['cv_folds'], shuffle=True, random_state=config['random_state'])

    rmse_scores = []
    mae_scores = []
    r2_scores = []

    fold = 1
    for train_index, val_index in kf.split(X):
        logger.info(f"Starting Fold {fold}")

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Кодування категоріальних змінних тільки на тренувальному наборі
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        if categorical_cols:
            one_hot_encoder = OneHotEncoder(cols=categorical_cols, use_cat_names=True)
            X_train_encoded = one_hot_encoder.fit_transform(X_train)
            X_val_encoded = one_hot_encoder.transform(X_val)
        else:
            X_train_encoded = X_train.copy()
            X_val_encoded = X_val.copy()

        # Масштабування даних
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_val_scaled = scaler.transform(X_val_encoded)

        # Побудова моделі
        model_params = {
            'input_dim': X_train_scaled.shape[1],
            'layers': config['model_layers'],
            'activation': config['activation'],
            'dropout_rates': config['dropout_rates'],
            'use_batch_norm': config['use_batch_norm'],
            'use_dropout': config['use_dropout'],  # Додано
            'learning_rate': config['learning_rate'],
            'l2_reg': config['l2_reg'],
            'resnet_blocks': config['resnet_blocks']
        }
        model = ModelBuilder.build_model(**model_params)

        # Навчання моделі
        history = ModelBuilder.train_model(model, X_train_scaled, y_train, X_val_scaled, y_val, config, fold)

        # Візуалізація історії навчання
        Visualization.plot_training_history(history, fold)

        # Оцінка моделі
        model_path = f'best_model_fold_{fold}.keras'
        rmse, mae, r2, y_val_exp, y_pred = ModelEvaluator.evaluate_model(model_path, X_val_scaled, y_val)

        logger.info(f'Fold {fold} Evaluation: RMSE: {rmse:.2f}, MAE: {mae:.2f}, R² Score: {r2:.4f}')

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        # Візуалізація результатів
        Visualization.plot_predictions(y_val_exp, y_pred, fold)
        Visualization.plot_residuals(y_val_exp, y_pred, fold)

        # Використання SHAP, якщо увімкнено
        if config.get('use_shap', False):
            Visualization.plot_feature_importance(model, X_train_encoded.columns, fold, use_shap=True,
                                                  X_train=X_train_scaled)
        else:
            Visualization.plot_feature_importance(model, X_train_encoded.columns, fold, use_shap=False)

        # Додаткова візуалізація розподілу ознак, якщо вказано
        plot_features = config.get('plot_features', [])
        for feature_name in plot_features:
            if feature_name in X_train_encoded.columns:
                feature_index = list(X_train_encoded.columns).index(feature_name)
                Visualization.plot_feature_distribution(X_train_scaled, X_val_scaled, feature_index, fold)
            else:
                logger.warning(f"Feature '{feature_name}' not found in encoded training data.")

        fold += 1

    # Середні метрики по всім фолдам
    logger.info(f'Average RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}')
    logger.info(f'Average MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}')
    logger.info(f'Average R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}')

    # Visualization.plot_correlation_matrix(X_encoded)


if __name__ == "__main__":
    main()
