import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("veri.csv")

kolonlar = {
    "id": "Kayıt numarası",
    "Gender": "Cinsiyet",
    "Age": "Yaş",
    "Height": "Boy",
    "Weight": "Kilo",
    "family_history_with_overweight": "Ailede fazla kilo geçmişi",
    "FAVC": "Yüksek kalorili yiyecek tüketimi",
    "FCVC": "Sebze tüketim sıklığı",
    "NCP": "Günlük ana öğün sayısı",
    "CAEC": "Öğün arası yeme alışkanlığı",
    "SMOKE": "Sigara kullanımı",
    "CH2O": "Günlük su tüketimi",
    "SCC": "Kalori takibi yapıyor mu",
    "FAF": "Fiziksel aktivite sıklığı",
    "TUE": "Teknoloji kullanım süresi",
    "CALC": "Alkol tüketimi",
    "MTRANS": "Ulaşım türü",
    "NObeyesdad": "Obezite seviyesi (hedef değişken)",
     "BMI": "vki"
}
print(" KOLON AÇIKLAMALARI:\n")
for col in df.columns:
    print(f"{col} ({kolonlar[col]})")
     
print("\n Eksik veri kontrolü:")
print(df.isnull().sum())

print("\n Veri tipleri:")
print(df.dtypes)


df["BMI"] = df["Weight"] / (df["Height"] ** 2)
df = df.drop(["id","Height", "Weight"], axis=1)

y = df["NObeyesdad"]
X = df.drop("NObeyesdad", axis=1)

print("\n Giriş Değişkenleri :")
for col in X.columns:
    print(col)

print(df["NObeyesdad"].value_counts())

X = pd.get_dummies(X)

le = LabelEncoder()
y = le.fit_transform(y)

print("\n Obezite Seviyeleri\n")
for i, sinif in enumerate(le.classes_):
    print(f"{i} → {sinif}")

#LOJİSTİK REGRESYON MODELİ
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

olceklendirici = StandardScaler()
X_egitim = olceklendirici.fit_transform(X_egitim)
X_test = olceklendirici.transform(X_test)

lojistik_model = LogisticRegression(max_iter=1000)
lojistik_model.fit(X_egitim, y_egitim)
y_tahmin = lojistik_model.predict(X_test)

dogruluk = accuracy_score(y_test, y_tahmin)
hassaslik = precision_score(y_test, y_tahmin, average="weighted")
duyarlilik = recall_score(y_test, y_tahmin, average="weighted")
f1 = f1_score(y_test, y_tahmin, average="weighted")

print("\n LOJİSTİK REGRESYON SONUÇLARI")
print("Doğruluk (Accuracy):", dogruluk)
print("Hassaslık (Precision):", hassaslik)
print("Duyarlılık (Recall):", duyarlilik)
print("F1 Skoru:", f1)

print("\n Sınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin))
print("\n Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_tahmin))

#RANDOM FOREST MODELİ 

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_egitim, y_egitim)
y_tahmin_rf = rf_model.predict(X_test)

dogruluk = accuracy_score(y_test, y_tahmin_rf)
hassaslik = precision_score(y_test, y_tahmin_rf, average="weighted")
duyarlilik = recall_score(y_test, y_tahmin_rf, average="weighted")
f1 = f1_score(y_test, y_tahmin_rf, average="weighted")

print("\n RANDOM FOREST SONUÇLARI")
print("Doğruluk (Accuracy):", dogruluk)
print("Hassaslık (Precision):", hassaslik)
print("Duyarlılık (Recall):", duyarlilik)
print("F1 Skoru:", f1)

print("\n Sınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_rf))

print("\n Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_tahmin_rf))


#KNN MODELİ

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_egitim, y_egitim)
y_tahmin_knn = knn_model.predict(X_test)
dogruluk = accuracy_score(y_test, y_tahmin_knn)
hassaslik = precision_score(y_test, y_tahmin_knn, average="weighted")
duyarlilik = recall_score(y_test, y_tahmin_knn, average="weighted")
f1 = f1_score(y_test, y_tahmin_knn, average="weighted")

print("\n KNN SONUÇLARI")
print("Doğruluk (Accuracy):", dogruluk)
print("Hassaslık (Precision):", hassaslik)
print("Duyarlılık (Recall):", duyarlilik)
print("F1 Skoru:", f1)

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_knn))

print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_tahmin_knn))


#SVM MODELİ,

svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_egitim, y_egitim)
y_tahmin_svm = svm_model.predict(X_test)
dogruluk = accuracy_score(y_test, y_tahmin_svm)
hassaslik = precision_score(y_test, y_tahmin_svm, average="weighted")
duyarlilik = recall_score(y_test, y_tahmin_svm, average="weighted")
f1 = f1_score(y_test, y_tahmin_svm, average="weighted")

print("\n SVM SONUÇLARI")
print("Doğruluk (Accuracy):", dogruluk)
print("Hassaslık (Precision):", hassaslik)
print("Duyarlılık (Recall):", duyarlilik)
print("F1 Skoru:", f1)

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_svm))

print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_tahmin_svm))


#KARAR AĞAÇLARI


karar_agaci_model = DecisionTreeClassifier(random_state=42)
karar_agaci_model.fit(X_egitim, y_egitim)
y_tahmin_karar_agaci = karar_agaci_model.predict(X_test)
dogruluk = accuracy_score(y_test, y_tahmin_karar_agaci)
hassaslik = precision_score(y_test, y_tahmin_karar_agaci, average="weighted")
duyarlilik = recall_score(y_test, y_tahmin_karar_agaci, average="weighted")
f1 = f1_score(y_test, y_tahmin_karar_agaci, average="weighted")

print("\n KARAR AĞACI SONUÇLARI")
print("Doğruluk (Accuracy):", dogruluk)
print("Hassaslık (Precision):", hassaslik)
print("Duyarlılık (Recall):", duyarlilik)
print("F1 Skoru:", f1)

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_karar_agaci))

print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_tahmin_karar_agaci))
    

#overfitting/underfitting durumu

print("\n OVERFITTING / UNDERFITTING KONTROLÜ\n")
print("Lojistik Regresyon")
print("Train:", lojistik_model.score(X_egitim, y_egitim))
print("Test :", lojistik_model.score(X_test, y_test))
print("\nKNN")
print("Train:", knn_model.score(X_egitim, y_egitim))
print("Test :", knn_model.score(X_test, y_test))
print("\nSVM")
print("Train:", svm_model.score(X_egitim, y_egitim))
print("Test :", svm_model.score(X_test, y_test))
print("\nRandom Forest")
print("Train:", rf_model.score(X_egitim, y_egitim))
print("Test :", rf_model.score(X_test, y_test))
print("\nKarar Ağacı")
print("Train:", karar_agaci_model.score(X_egitim, y_egitim))
print("Test :", karar_agaci_model.score(X_test, y_test))


#lazypredict
'''clf = LazyClassifier(verbose=0, ignore_warnings=True)
modeller, tahminler = clf.fit(X_egitim, X_test, y_egitim, y_test)
print(modeller)'''



#xgb


xgb_model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

xgb_model.fit(X_egitim, y_egitim)
y_tahmin_xgb = xgb_model.predict(X_test)

dogruluk = accuracy_score(y_test, y_tahmin_xgb)
hassaslik = precision_score(y_test, y_tahmin_xgb, average="weighted")
duyarlilik = recall_score(y_test, y_tahmin_xgb, average="weighted")
f1 = f1_score(y_test, y_tahmin_xgb, average="weighted")

print("\n XGBOOST SONUÇLARI")
print("Doğruluk (Accuracy):", dogruluk)
print("Hassaslık (Precision):", hassaslik)
print("Duyarlılık (Recall):", duyarlilik)
print("F1 Skoru:", f1)

print("\n Sınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_xgb))

print("\n Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_tahmin_xgb))


print("\n XGBOOST OVERFITTING / UNDERFITTING KONTROLÜ")
print("Train:", xgb_model.score(X_egitim, y_egitim))
print("Test :", xgb_model.score(X_test, y_test))


#bagging

bagging_model = BaggingClassifier(random_state=42)
bagging_model.fit(X_egitim, y_egitim)
y_tahmin_bag = bagging_model.predict(X_test)
dogruluk = accuracy_score(y_test, y_tahmin_bag)
hassaslik = precision_score(y_test, y_tahmin_bag, average="weighted")
duyarlilik = recall_score(y_test, y_tahmin_bag, average="weighted")
f1 = f1_score(y_test, y_tahmin_bag, average="weighted")

print("\n BAGGING SONUÇLARI")
print("Doğruluk (Accuracy):", dogruluk)
print("Hassaslık (Precision):", hassaslik)
print("Duyarlılık (Recall):", duyarlilik)
print("F1 Skoru:", f1)

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_bag))

print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_tahmin_bag))

print("\n BAGGING OVERFITTING KONTROLÜ")
print("Train:", bagging_model.score(X_egitim, y_egitim))
print("Test :", bagging_model.score(X_test, y_test))

#extra tree
extra_model = ExtraTreesClassifier(random_state=42)
extra_model.fit(X_egitim, y_egitim)
y_tahmin_extra = extra_model.predict(X_test)

dogruluk = accuracy_score(y_test, y_tahmin_extra)
hassaslik = precision_score(y_test, y_tahmin_extra, average="weighted")
duyarlilik = recall_score(y_test, y_tahmin_extra, average="weighted")
f1 = f1_score(y_test, y_tahmin_extra, average="weighted")


print("\n EXTRA TREES SONUÇLARI")
print("Doğruluk (Accuracy):", dogruluk)
print("Hassaslık (Precision):", hassaslik)
print("Duyarlılık (Recall):", duyarlilik)
print("F1 Skoru:", f1)

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_extra))

print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_tahmin_extra))

print("\n EXTRA TREES OVERFITTING KONTROLÜ")
print("Train:", extra_model.score(X_egitim, y_egitim))
print("Test :", extra_model.score(X_test, y_test))



#en iyi model olan xgbyi iyileştirme

xgb_parametreler = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    estimator=XGBClassifier(
        random_state=42,
        eval_metric="mlogloss"
    ),
    param_grid=xgb_parametreler,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)
xgb_grid.fit(X_egitim, y_egitim)
en_iyi_xgb = xgb_grid.best_estimator_
y_tahmin_xgb_tuned = en_iyi_xgb.predict(X_test)
print("\n XGBOOST İYİLEŞTİRİLMİŞ SONUÇLAR")
print("Train Accuracy:", en_iyi_xgb.score(X_egitim, y_egitim))
print("Test Accuracy :", en_iyi_xgb.score(X_test, y_test))
print("\nAccuracy:", accuracy_score(y_test, y_tahmin_xgb_tuned))
print("Precision:", precision_score(y_test, y_tahmin_xgb_tuned, average="weighted"))
print("Recall:", recall_score(y_test, y_tahmin_xgb_tuned, average="weighted"))
print("F1 Score:", f1_score(y_test, y_tahmin_xgb_tuned, average="weighted"))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_tahmin_xgb_tuned))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test, y_tahmin_xgb_tuned))
# model eğitimi dosyana ekle
import shap
import joblib

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

joblib.dump(shap_values, "shap_values.pkl")
joblib.dump(X_test, "X_test_shap.pkl")