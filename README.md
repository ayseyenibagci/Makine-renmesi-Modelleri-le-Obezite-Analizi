Bu projede kullanılan veri seti :
https://www.kaggle.com/datasets/adeniranstephen/obesity-prediction-dataset/data?utm_source

# Obezite Seviyesi Tahmin Uygulaması

Bu proje, bireylerin fiziksel özellikleri, beslenme alışkanlıkları ve yaşam tarzı bilgilerini kullanarak obezite seviyelerini makine öğrenmesi yöntemleriyle tahmin etmek amacıyla geliştirilmiştir.

Proje kapsamında farklı makine öğrenmesi algoritmaları karşılaştırılmış, en başarılı modellerden biri olan XGBoost üzerinde hiperparametre optimizasyonu gerçekleştirilmiş ve elde edilen model Streamlit tabanlı bir web uygulamasında kullanılmıştır.

> Not: Bu uygulama eğitim ve makine öğrenmesi projesi amacıyla geliştirilmiştir. Üretilen sonuçlar tıbbi tanı veya profesyonel sağlık değerlendirmesi yerine geçmez.

---

## Projenin Amacı

Projenin temel amacı, kişilerin yaşam tarzı ve fiziksel özelliklerinden yararlanarak obezite seviyelerini otomatik olarak sınıflandırabilen bir makine öğrenmesi modeli geliştirmektir.

Model aşağıdaki gibi bilgileri kullanmaktadır:

* Cinsiyet
* Yaş
* Boy
* Kilo
* Ailede fazla kilo geçmişi
* Yüksek kalorili yiyecek tüketimi
* Sebze tüketim sıklığı
* Günlük ana öğün sayısı
* Öğün arası yeme alışkanlığı
* Sigara kullanımı
* Günlük su tüketimi
* Kalori takibi
* Fiziksel aktivite sıklığı
* Teknoloji kullanım süresi
* Alkol tüketimi
* Ulaşım türü
* Vücut Kitle İndeksi (BMI)

---

## Veri Seti

Projede kullanılan veri seti `veri.csv` dosyasında bulunmaktadır.

Veri setindeki hedef değişken:

```text
NObeyesdad
```

Hedef değişken toplam 7 farklı obezite sınıfından oluşmaktadır:

| Kod                 | Türkçe Karşılığı     |
| ------------------- | -------------------- |
| Insufficient_Weight | Yetersiz Kilo        |
| Normal_Weight       | Normal Kilo          |
| Overweight_Level_I  | Fazla Kilo Seviye I  |
| Overweight_Level_II | Fazla Kilo Seviye II |
| Obesity_Type_I      | Obezite Tip I        |
| Obesity_Type_II     | Obezite Tip II       |
| Obesity_Type_III    | Obezite Tip III      |

---

## BMI Hesaplama

Projede boy ve kilo bilgilerinden Vücut Kitle İndeksi (BMI) hesaplanmaktadır.

Kullanılan formül:

```text
BMI = Kilo / Boy²
```

Boy değeri metre cinsinden kullanılmaktadır.

Örneğin:

```text
Kilo = 65 kg
Boy = 1.65 m

BMI = 65 / (1.65²)
```

Hesaplanan BMI değeri modelde giriş değişkenlerinden biri olarak kullanılmaktadır.

---

## Veri Ön İşleme

Model eğitimi öncesinde veri üzerinde çeşitli ön işleme işlemleri uygulanmıştır.

Uygulanan işlemler:

1. Veri setinin CSV dosyasından okunması
2. Eksik veri kontrolü
3. BMI değişkeninin oluşturulması
4. `id`, `Height` ve `Weight` değişkenlerinin model girdisinden çıkarılması
5. Kategorik değişkenlerin One-Hot Encoding yöntemiyle dönüştürülmesi
6. Hedef değişkenin `LabelEncoder` ile sayısal değerlere dönüştürülmesi
7. Verinin eğitim ve test olarak ayrılması
8. Sayısal verilerin `StandardScaler` ile ölçeklendirilmesi

Veri:

```text
%80 Eğitim
%20 Test
```

şeklinde ayrılmış ve `random_state=42` kullanılmıştır.

---

## Kullanılan Makine Öğrenmesi Modelleri

Projede birden fazla sınıflandırma algoritması denenmiştir:

* Logistic Regression
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* XGBoost
* Bagging Classifier
* Extra Trees Classifier

Ayrıca modellerin karşılaştırılması amacıyla LazyPredict kütüphanesinin kullanılması planlanmıştır.

---

## XGBoost

Projede ana model olarak XGBoost kullanılmıştır.

XGBoost'un tercih edilme nedenleri:

* Tabular verilerde güçlü performans göstermesi
* Çok sınıflı sınıflandırma problemlerine uygun olması
* Doğrusal olmayan ilişkileri öğrenebilmesi
* Özellik önemlerinin incelenebilmesi
* Düzenlileştirme mekanizmaları sayesinde overfitting riskinin azaltılabilmesi

Model:

```python
XGBClassifier(
    random_state=42,
    eval_metric="mlogloss"
)
```

şeklinde oluşturulmuştur.

---

## Hiperparametre Optimizasyonu

XGBoost modelinin performansını geliştirmek amacıyla GridSearchCV kullanılmıştır.

Araştırılan hiperparametreler:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
```

5 katlı çapraz doğrulama (`cv=5`) uygulanmıştır.

Model seçiminde:

```text
F1 Weighted
```

metriği kullanılmıştır.

En iyi parametre kombinasyonu GridSearchCV tarafından belirlenerek geliştirilmiş XGBoost modeli oluşturulmuştur.

---

## Model Değerlendirme

Modeller aşağıdaki performans metrikleri kullanılarak değerlendirilmiştir:

### Accuracy

Modelin tüm tahminler içerisindeki doğru tahmin oranını gösterir.

### Precision

Modelin belirli bir sınıf olarak tahmin ettiği örneklerin ne kadarının gerçekten o sınıfa ait olduğunu gösterir.

### Recall

Gerçek bir sınıfa ait örneklerin ne kadarının doğru şekilde yakalandığını gösterir.

### F1 Score

Precision ve Recall değerlerinin harmonik ortalamasıdır.

Projede çok sınıflı yapı nedeniyle metriklerde:

```python
average="weighted"
```

kullanılmıştır.

---

## Kullanılan Analizler

Projenin Streamlit arayüzünde çeşitli veri ve model analizleri sunulmaktadır.

### Obezite Sınıf Dağılımı

Veri setindeki 7 obezite sınıfının dağılımı görselleştirilmektedir.

### Korelasyon Matrisi

Sayısal değişkenler arasındaki ilişkiler incelenmektedir.

İncelenen değişkenlerden bazıları:

* BMI
* Yaş
* Fiziksel aktivite
* Su tüketimi
* Teknoloji kullanım süresi
* Sebze tüketimi
* Ana öğün sayısı

### Cramér's V Analizi

Kategorik değişkenler arasındaki ilişkinin incelenmesi amacıyla Cramér's V kullanılmıştır.

### XGBoost Özellik Önemi

Modelin tahminlerinde hangi özelliklerin daha fazla katkı sağladığı incelenmektedir.

### Karışıklık Matrisi

Modelin sınıflar bazındaki doğru ve yanlış tahminleri görselleştirilmektedir.

### ROC Eğrisi

Her sınıf için ROC eğrileri oluşturularak AUC değerleri gösterilmektedir.

### PCA

Test verileri iki boyutlu PCA uzayına indirgenerek gerçek ve tahmin edilen sınıfların dağılımları görselleştirilmektedir.

### SHAP Analizi

SHAP kullanılarak özelliklerin model tahminleri üzerindeki etkisi incelenmektedir.

---

## Streamlit Uygulaması

Proje, kullanıcıların kendi bilgilerini girerek obezite seviyesi tahmini alabilecekleri bir Streamlit web uygulamasına dönüştürülmüştür.

Uygulamada üç temel bölüm bulunmaktadır:

### Ana Sayfa

Projenin amacı ve kullanılan XGBoost modeli hakkında bilgi verilmektedir.

### Tahmin Ekranı

Kullanıcıdan aşağıdaki bilgiler alınır:

* Cinsiyet
* Yaş
* Boy
* Kilo
* Ailede fazla kilo geçmişi
* Yüksek kalorili yiyecek tüketimi
* Sebze tüketimi
* Ana öğün sayısı
* Öğün arası yeme
* Sigara kullanımı
* Su tüketimi
* Kalori takibi
* Fiziksel aktivite
* Teknoloji kullanımı
* Alkol tüketimi
* Ulaşım türü

Boy ve kilo bilgilerinden BMI otomatik olarak hesaplanır.

Ardından eğitilmiş XGBoost modeli kullanılarak obezite seviyesi tahmin edilir.

### EDA Analizleri

Bu bölümde:

* Sınıf dağılımı
* Korelasyon matrisi
* Cramér's V
* XGBoost özellik önemi
* Karışıklık matrisi
* PCA
* ROC eğrisi
* SHAP analizi

gösterilmektedir.

---

## Proje Dosya Yapısı

```text
proje/
│
├── veri.csv
├── model_egitimi.py
├── app.py
│
├── xgboost_obezite_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── target_encoder.pkl
│
├── X_test_scaled.pkl
├── y_test.pkl
├── y_pred_xgb.pkl
├── shap_values.pkl
├── X_test_shap.pkl
│
├── proje.png
├── proje2.png
│
├── requirements.txt
└── README.md
```

> Dosya isimleri, projede kullanılan gerçek dosya adlarıyla aynı tutulmalıdır.

---

## Kullanılan Teknolojiler

Proje geliştirilirken aşağıdaki Python kütüphanelerinden yararlanılmıştır:

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LazyPredict
* SHAP
* Matplotlib
* Streamlit
* Joblib
* Pillow
* SciPy

---

## Kurulum

Projeyi bilgisayarınıza klonladıktan sonra gerekli kütüphaneleri yükleyebilirsiniz.

```bash
pip install -r requirements.txt
```

Streamlit uygulamasını çalıştırmak için:

```bash
streamlit run app.py
```

Ardından Streamlit tarafından verilen yerel adresten uygulamaya erişebilirsiniz.

---

## Proje Akışı

```text
Veri Seti
    ↓
Veri Ön İşleme
    ↓
BMI Hesaplama
    ↓
Kategorik Değişkenlerin Kodlanması
    ↓
Train / Test Ayrımı
    ↓
StandardScaler
    ↓
Farklı Makine Öğrenmesi Modelleri
    ↓
Model Karşılaştırması
    ↓
XGBoost Seçimi
    ↓
GridSearchCV ile Optimizasyon
    ↓
Model Değerlendirme
    ↓
SHAP / PCA / ROC / Confusion Matrix
    ↓
Streamlit Web Uygulaması
    ↓
Obezite Seviyesi Tahmini
```

---

## Sonuç

Bu proje kapsamında bireylerin fiziksel özellikleri, beslenme alışkanlıkları ve yaşam tarzı bilgileri kullanılarak 7 farklı obezite seviyesi sınıflandırılmaya çalışılmıştır.

Birden fazla makine öğrenmesi algoritması karşılaştırılmış ve XGBoost modeli üzerinde hiperparametre optimizasyonu uygulanmıştır.

Geliştirilen Streamlit uygulaması sayesinde kullanıcılar kendi bilgilerini girerek modelin tahmin ettiği obezite seviyesini görebilmekte ve girilen bilgilere göre çeşitli genel sağlık önerileri sunulmaktadır.

Ayrıca modelin nasıl karar verdiğini daha iyi anlayabilmek amacıyla özellik önemi, SHAP, ROC, PCA ve karışıklık matrisi gibi analizlerden yararlanılmıştır.

---

## Proje

Bu çalışma, makine öğrenmesi ve veri analizi uygulamalarını öğrenmek ve pratikte kullanmak amacıyla geliştirilmiştir.
