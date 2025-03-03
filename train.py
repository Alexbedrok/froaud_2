import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random

# Количество строк в датасете
num_samples = 1000

# Генерация случайных данных
user_ids = [f"user_{i}" for i in range(1, 21)]  # 20 уникальных пользователей
merchants = ['Electronics', 'Groceries', 'Clothing', 'Travel', 'Entertainment']

# Случайные транзакции
data = {
    'Amount': np.random.uniform(5, 1000, size=num_samples),
    'TransactionTime': pd.date_range('2025-01-01', periods=num_samples, freq='T'),
    'Merchant': [random.choice(merchants) for _ in range(num_samples)],
    'UserID': [random.choice(user_ids) for _ in range(num_samples)],
    'IsFraud': [random.choices([0, 1], weights=[0.95, 0.05])[0] for _ in range(num_samples)]  # 5% мошенничества
}

df = pd.DataFrame(data)

# Преобразуем категориальные данные (Merchant, UserID) в числовые значения
label_encoder = LabelEncoder()
df['Merchant'] = label_encoder.fit_transform(df['Merchant'])
df['UserID'] = label_encoder.fit_transform(df['UserID'])

# Сохраняем данные в CSV
df.to_csv('transactions.csv', index=False)

print("CSV файл для тренировки создан!")
