import sys

import numpy as np
import onnxruntime as rt
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

drone_csvs = sys.argv[1].split(",")
drone_df = pd.concat(
    [pd.read_csv(csv, header=None) for csv in drone_csvs], axis=0, ignore_index=True
)

y = [1 for _ in range(len(drone_df))]

bg_csvs = sys.argv[2].split(",")
bg_df = pd.concat(
    [pd.read_csv(csv, header=None) for csv in bg_csvs], axis=0, ignore_index=True
)

y.extend([0 for _ in range(len(bg_df))])

X = pd.concat([drone_df, bg_df], axis=0, ignore_index=True)
X = X.iloc[:, 1:]  # for testing with location csvs, until we have proper detection csvs
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=16, random_state=42, n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Acc: {acc}")

onx = to_onnx(
    rf,
    # X_train,
    # X.iloc[0].to_numpy(),
    initial_types=[("input", FloatTensorType([None, 682]))],
    # final_types=[("variable", Int32TensorType([None, 1]))],
    # options={"zipmap": False},
)
with open(sys.argv[-1], "wb") as f:
    f.write(onx.SerializeToString())

# Compute the prediction with onnxruntime.
import onnxruntime as rt

sess = rt.InferenceSession(sys.argv[-1], providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print(f"input_name: {input_name} | label_name: {label_name}")
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
print(pred_onx)
