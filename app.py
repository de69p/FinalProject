import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from nn2 import ConfigLoader, DataPreprocessor, ModelBuilder, ModelEvaluator, Visualization
from vis import (
    plot_price_distribution,
    plot_real_prices_per_district,
    plot_price_vs_area,
    plot_price_vs_distance
)

st.title("Real Estate Price Prediction")
st.write("""
    У цьому проєкті аналізуються та прогнозуються ціни на нерухомість на основі даних об'єктів у Києві.
    Процес включає кілька етапів: дослідження розподілу цін, аналіз факторів, що впливають на ціну,
    та побудову моделі для прогнозування цін на основі наявних даних.
""")

data_path = 'dataset.csv'
data = pd.read_csv(data_path)
st.write("Data Loaded Automatically")
st.dataframe(data.head())

st.subheader("Price Distribution")
st.write("Цей графік показує розподіл цін на нерухомість. Він допомагає виявити аномально високі або низькі ціни."
         "Графік показує, що більшість об'єктів нерухомості мають ціну до 500 тисяч доларів."
         "Розподіл сильно знижується на вищих цінових рівнях, а ціни понад 1 мільйон доларів зустрічаються рідко,"
         "що свідчить про обмежену кількість елітного житла на ринку.")
fig1 = plot_price_distribution(data)
st.plotly_chart(fig1)

st.subheader("Ratio of prices by district")
st.write("""
    Графік відображає всі ціни на нерухомість у різних районах міста. Кожна точка представляє окремий об'єкт нерухомості з його ціною.
    У Печерському та Шевченківському районах спостерігаються найбільш високі ціни, що видно за скупченням точок на графіку.
    Також можна побачити, що деякі точки значно виходять за межі звичайного рівня цін, що вказує на наявність елітного житла.
    Особливо це помітно в Печерському районі, де ціни можуть перевищувати 4 мільйони доларів.
""")
fig2 = plot_real_prices_per_district(data)
st.plotly_chart(fig2)

st.subheader("Price vs Area")
st.write(
    "На графіку видно, що більшість нерухомості має площу до 1000 квадратних метрів, а ціни варіюються в межах до 4 мільйонів доларів."
    "На графіку є один аутлаєр із великою площею, який, скоріш за все, є некоректним або фіктивним значенням.")
fig3 = plot_price_vs_area(data)
st.plotly_chart(fig3)

st.subheader("Price vs Distance to Center")
st.write(
    "Графік показує залежність цін на нерухомість від відстані до центру міста."
    "Найвищі ціни спостерігаються поблизу центру, особливо до 6 км."
    "На відстані понад 10 км ціни значно знижуються, а об’єкти стають більш доступними, незалежно від району.")
fig4 = plot_price_vs_distance(data)
st.plotly_chart(fig4)

# Тренування моделі
if st.button("Train Model"):
    # Обробка даних для аналізу
    data = DataPreprocessor.feature_engineering(data)
    config = ConfigLoader.load_config()
    data_encoded = DataPreprocessor.encode_features(data)
    X_train, X_test, y_train, y_test = DataPreprocessor.prepare_data(data_encoded, config)

    # Параметри моделі
    model_params = {
        'input_dim': X_train.shape[1],
        'layers': config['model_layers'],
        'activation': config['activation'],
        'dropout_rates': config['dropout_rates'],
        'use_batch_norm': config['use_batch_norm'],
        'learning_rate': config['learning_rate'],
        'l2_reg': config['l2_reg'],
        'resnet_blocks': config['resnet_blocks']
    }

    # Архітектура мережі
    st.subheader("Model Architecture")
    st.write("""
            Модель складається з трьох прихованих шарів з кількістю нейронів: 256, 128, і 64. В кожному шарі використовується
            функція активації Swish, що підвищує продуктивність на задачах регресії. Також застосовується нормалізація пакетів
            (Batch Normalization) і регуляризація Dropout з коефіцієнтами 0.3, 0.2 і 0.1 для підвищення стабільності навчання.
            Для боротьби з проблемою зникнення градієнтів використовуються ResNet-блоки, які дозволяють зберегти контекст попередніх
            шарів через skip-зв'язки. Нейронна мережа оптимізується за допомогою Adam з функцією втрат Log-Cosh і регуляризацією L2
            для запобігання перенавчанню.
        """)

    model = ModelBuilder.build_model(**model_params)

    # Налаштування прогресу
    progress_bar = st.progress(0)
    status_text = st.empty()


    def on_epoch_end(epoch, logs):
        # Оновлюємо прогрес для кожної епохи
        progress = (epoch + 1) / config['epochs']
        progress_bar.progress(progress)  # Оновлення прогрес-бара

        # Додаємо метрики до статусного тексту
        status_text.text(
            f"Epoch {epoch + 1}/{config['epochs']} - "
            f"Loss: {logs['loss']:.4f}, "
            f"Val Loss: {logs['val_loss']:.4f}, "
            f"MAE: {logs.get('mae', 'N/A'):.4f}, "
            f"Val MAE: {logs.get('val_mae', 'N/A'):.4f}"
        )


    # Додавання колбека
    custom_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

    # Початок тренування моделі
    history = model.fit(
        X_train, y_train,
        validation_split=config['validation_split'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=[custom_callback, *ModelBuilder.get_callbacks()],
        verbose=0
    )

    st.success('Model training completed!')

    # Оцінюємо модель на тестовому наборі
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    st.write(f"Test Loss: {test_loss:.4f}")
    st.write(f"Test MAE: {test_mae:.4f}")

    # Розрахунок інших метрик для тестового набору
    y_test_exp, y_pred = ModelEvaluator.evaluate_model('best_model.keras', X_test, y_test)

    rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred))
    mae = np.mean(np.abs(y_test_exp - y_pred))  # MAE в доларах
    r2 = r2_score(y_test_exp, y_pred)

    # Вивід інших метрик
    st.write(f"Test RMSE: {rmse:.4f}")
    st.write(f"Test R² Score: {r2:.4f}")

    # Графік історії тренування
    st.subheader("Training and Validation Loss")
    st.write(
        "Графік показує зміну втрат на тренувальному та валідаційному наборах.\n"
        "Спостерігається швидке зниження втрат на початку навчання, що свідчить про те,\n"
        "що модель швидко навчається розпізнавати основні патерни у даних. Згодом крива втрат стабілізується,\n"
        "що означає, що модель досягає рівноваги між навчанням і узагальненням.")
    final_fig = Visualization.plot_training_history(history.history)
    st.plotly_chart(final_fig)

    # Оцінка моделі
    y_test_exp, y_pred = ModelEvaluator.evaluate_model('best_model.keras', X_test, y_test)

    st.subheader("Actual vs Predicted Prices")
    st.write(
        "Графік порівнює реальні та передбачені ціни, дозволяючи оцінити точність моделі.\n"
        "Точки, що лежать близько до червоної пунктирної лінії, свідчать про високу точність,\n"
        "адже передбачені значення близькі до фактичних. Загалом, модель демонструє хорошу узгодженість з реальними даними,\n"
        "особливо для цін нижчого та середнього діапазонів. Для деяких високих значень можна помітити деякі відхилення від лінії,\n"
        "що вказує на можливі труднощі моделі з передбаченням екстремальних значень.")
    fig6 = Visualization.plot_predictions(y_test_exp, y_pred)
    st.plotly_chart(fig6)

    # Побудова таблиці порівняння фактичних та передбачених значень
    st.subheader("Comparison Table of Actual vs Predicted Prices")
    comparison_df = pd.DataFrame({
        'Actual Prices': y_test_exp,
        'Predicted Prices': y_pred
    })
    st.dataframe(comparison_df.head(20))

    # Побудова графіка важливості ознак
    st.subheader("Feature Importance")
    st.write(
        "Цей графік показує важливість різних ознак для моделі, що допомагає краще зрозуміти, які фактори найбільше впливають на ціну. "
        "Найбільший вплив на визначення ціни мають такі ознаки:\n\n"
        "- **Price_Distance_Interaction** — показує, як ціна залежить від відстані до центру міста. Вона суттєво впливає на вартість.\n"
        "- **Rooms_Area_Interaction** — враховує кількість кімнат і площу. Чим більше кімнат і площа, тим вища може бути ціна.\n"
        "- **Room_Class_medium** — середній клас кімнат. Важливий, бо показує популярний тип квартир.\n"
        "- **District_Печерський** — район Печерський значно впливає на ціну, адже це один із найбільш престижних районів.\n"
        "- **Room_Class_small** — мала кількість кімнат також важлива, бо такі квартири зазвичай коштують дешевше, але популярні серед певних покупців."
    )
    feature_names = data_encoded.drop(
        columns=['Price_USD', 'Log_Price', 'PricePerSqm_USD', 'Description', 'DateCreated', 'ID']).columns
    fig7 = Visualization.plot_feature_importance(model, feature_names)
    st.plotly_chart(fig7)
