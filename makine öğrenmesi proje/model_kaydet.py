import pandas as pd
import joblib
import shap

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier


df = pd.read_csv("veri.csv")

df["BMI"] = df["Weight"] / (df["Height"] ** 2)
df = df.drop(["id", "Height", "Weight"], axis=1)

y = df["NObeyesdad"]
X = df.drop("NObeyesdad", axis=1)

X = pd.get_dummies(X)

feature_names = X.columns.tolist()

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

X_egitim, X_test, y_egitim, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_egitim_scaled = scaler.fit_transform(X_egitim)
X_test_scaled = scaler.transform(X_test)

X_egitim_scaled_df = pd.DataFrame(X_egitim_scaled, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss"
)

xgb_model.fit(X_egitim_scaled_df, y_egitim)

y_tahmin_xgb = xgb_model.predict(X_test_scaled_df)

print("\nXGBOOST SONUÇLARI")
print("Accuracy:", accuracy_score(y_test, y_tahmin_xgb))
print("Precision:", precision_score(y_test, y_tahmin_xgb, average="weighted"))
print("Recall:", recall_score(y_test, y_tahmin_xgb, average="weighted"))
print("F1 Score:", f1_score(y_test, y_tahmin_xgb, average="weighted"))

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_xgb))

print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_tahmin_xgb))

print("\nXGBOOST OVERFITTING / UNDERFITTING KONTROLÜ")
print("Train:", xgb_model.score(X_egitim_scaled_df, y_egitim))
print("Test :", xgb_model.score(X_test_scaled_df, y_test))


joblib.dump(xgb_model, "xgboost_obezite_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

joblib.dump(X_test_scaled_df, "X_test_scaled.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(y_tahmin_xgb, "y_pred_xgb.pkl")

joblib.dump(X_egitim_scaled_df, "X_train_scaled.pkl")
joblib.dump(y_egitim, "y_train.pkl")


explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled_df)

joblib.dump(shap_values, "shap_values.pkl")
joblib.dump(X_test_scaled_df, "X_test_shap.pkl")

print("\nTüm model ve test dosyaları başarıyla kaydedildi.")