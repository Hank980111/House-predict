import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 假資料（你可以用真實資料取代）
data = {
    '面積': [30, 45, 60, 80, 100, 120, 150, 200],
    '房間數': [1, 1, 2, 2, 3, 3, 4, 5],
    '地點編號': [1, 2, 1, 2, 3, 3, 4, 5],  # 假設地點轉成編號
    '房價': [300, 450, 600, 800, 1000, 1200, 1500, 2000]
}

# 將資料轉為 DataFrame
df = pd.DataFrame(data)

# 特徵與標籤
X = df[['面積', '房間數', '地點編號']]
y = df['房價']

# 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 顯示結果
print("預測房價：", y_pred)
print("實際房價：", list(y_test))
print("均方誤差（MSE）：", mean_squared_error(y_test, y_pred))

# 示範：預測新房子
new_house = pd.DataFrame([[110, 3, 3]], columns=['面積', '房間數', '地點編號'])
predicted_price = model.predict(new_house)
print("新房子的預測房價：", predicted_price[0])