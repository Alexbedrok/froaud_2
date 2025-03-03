import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Загрузка и подготовка данных
data = pd.read_csv('transactions.csv')

# Предположим, что данные содержат колонку 'is_fraud' (метка: мошенничество/не мошенничество)
X = data.drop(columns=['is_fraud'])
y = data['is_fraud']

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающие и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Создание модели нейросети
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Для бинарной классификации

# Компиляция модели
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Оценка точности модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')
