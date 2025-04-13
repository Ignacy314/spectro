import sys

import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import (DoubleTensorType, FloatTensorType,
                                        Int64TensorType)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data_csvs = sys.argv[1:-1]
dfs = [pd.read_csv(csv, header=None) for csv in data_csvs]

X = pd.concat([df.iloc[:, 1:] for df in dfs], axis=0, ignore_index=True)
y = pd.concat([df.iloc[:, 1] for df in dfs], axis=0, ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print(y_test)

rf = RandomForestRegressor(random_state=42, n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse} | R2: {r2}")

# onx = skl2onnx.convert_sklearn(
#     rf,
#     initial_types=[("input", FloatTensorType([None, 682]))],
#     final_types=[("variable", FloatTensorType([None, 1]))],
#     # options={'zipmap': False},
# )
# onx = to_onnx(rf, X[:1].astype(np.float32), options={"zipmap": False})
onx = to_onnx(
    rf,
    # X.iloc[0].to_numpy(),
    initial_types=[("input", FloatTensorType([None, 682]))],
    final_types=[("variable", DoubleTensorType([None, 1]))],
    # options={"zipmap": False},
)
with open(sys.argv[-1], "wb") as f:
    f.write(onx.SerializeToString())
