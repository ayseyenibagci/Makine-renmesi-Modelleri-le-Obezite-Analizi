import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="Obezite Seviyesi Tahmin Uygulaması",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarNav"],
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] div {
    background-color: #FFFFFF !important;
}

[data-testid="stSidebar"] * {
    color: #000000 !important;
}

.stRadio label {
    background:#f5f5f5 !important;
    padding: 10px 14px;
    border-radius: 12px;
    border: 1px solid #dcdcdc !important;
    display:flex !important;
    align-items:center;
    justify-content:flex-start;
    width:100%;
    gap:8px;
    cursor:pointer;
}

.stRadio label p {
    color:black !important;
    font-weight:700 !important;
    white-space:nowrap !important;
    overflow:hidden !important;
    text-overflow:ellipsis !important;
}

.stRadio label:hover {
    box-shadow:0 0 0 1px #800020 inset;
}

.stRadio input:checked + div {
    background:#800020 !important;
    color:white !important;
    border-radius:12px;
    box-shadow:none !important;
}

[data-testid="stToolbar"],
[data-testid="stDeployButton"],
[data-testid="baseButton-header"],
[data-testid="stHamburgerMenu"] {
    display:none !important;
}

header { 
    visibility:hidden !important; 
}
</style>
""", unsafe_allow_html=True)


sinif_turkce = {
    "Insufficient_Weight": "Yetersiz Kilo",
    "Normal_Weight": "Normal Kilo",
    "Overweight_Level_I": "Fazla Kilo Seviye I",
    "Overweight_Level_II": "Fazla Kilo Seviye II",
    "Obesity_Type_I": "Obezite Tip I",
    "Obesity_Type_II": "Obezite Tip II",
    "Obesity_Type_III": "Obezite Tip III"
}


@st.cache_resource
def modeli_yukle():
    model = joblib.load("xgboost_obezite_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    return model, scaler, feature_names, target_encoder


model, scaler, feature_names, target_encoder = modeli_yukle()


@st.cache_data
def veriyi_hazirla():
    df = pd.read_csv("veri.csv")

    if "BMI" not in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # Model eğitim dosyasında kaydedilen gerçek test verileri kullanılır.
    # Böylece Streamlit içinde tekrar train_test_split yapılmaz.
    X_test_scaled = joblib.load("X_test_scaled.pkl")
    y_test = joblib.load("y_test.pkl")
    y_pred_xgb = joblib.load("y_pred_xgb.pkl")

    return df, X_test_scaled, y_test, y_pred_xgb


df_eda_global, X_test_scaled, y_test, y_pred_xgb = veriyi_hazirla()


menu = st.sidebar.radio(
    "MENÜ",
    ["Ana Sayfa", "Tahmin Ekranı", "EDA Analizleri"]
)

if menu == "Ana Sayfa":

    st.title("Obezite Seviyesi Tahmin Uygulaması")

    st.markdown("""
    <div style="font-weight:700; font-size:19px; line-height:1.7;">
    Bu projede bireyin <b>yaşam tarzı</b>, <b>fiziksel özellikleri</b> ve 
    <b>beslenme alışkanlıkları</b> kullanılarak obezite seviyesi tahmin edilmektedir.
    <br><br>
    Ana model olarak <b>XGBoost</b> kullanılmıştır. Model; yaş, cinsiyet,
    ailede fazla kilo geçmişi, yüksek kalorili yiyecek tüketimi, sebze tüketimi,
    öğün sayısı, su tüketimi, fiziksel aktivite, teknoloji kullanım süresi ve BMI
    gibi değişkenleri kullanarak obezite sınıfını tahmin eder.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    try:
        img1 = Image.open("proje.png")
        img2 = Image.open("proje2.png")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img1, use_container_width=True)

        with col2:
            st.image(img2, use_container_width=True)

    except FileNotFoundError:
        st.warning("Ana sayfa görselleri bulunamadı.")

    st.markdown("""
    <div style="font-weight:700; font-size:18px; line-height:1.7;">
    <br>
    <b>XGBoost modelinin seçilme nedenleri:</b>
    <ul>
        <li>Tabular veri setlerinde güçlü performans göstermesi</li>
        <li>Çok sınıflı sınıflandırma problemlerine uygun olması</li>
        <li>Doğrusal olmayan ilişkileri öğrenebilmesi</li>
        <li>Overfitting riskini azaltmaya yardımcı düzenlileştirme yapısı</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


elif menu == "Tahmin Ekranı":

    st.header("XGBoost ile Obezite Seviyesi Tahmini")
    st.markdown("Aşağıdaki bilgileri girerek tahmin alabilirsin.")

    sinif_sirasi = [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III"
    ]

    gender_map = {"Kadın": "Female", "Erkek": "Male"}
    evet_hayir_map = {"Evet": "yes", "Hayır": "no"}

    caec_map = {
        "Hayır": "no",
        "Bazen": "Sometimes",
        "Sık sık": "Frequently",
        "Her zaman": "Always"
    }

    calc_map = {
        "Hayır": "no",
        "Bazen": "Sometimes",
        "Sık sık": "Frequently",
        "Her zaman": "Always"
    }

    mtrans_map = {
        "Toplu taşıma": "Public_Transportation",
        "Yürüyüş": "Walking",
        "Otomobil": "Automobile",
        "Motosiklet": "Motorbike",
        "Bisiklet": "Bike"
    }

    sebze_secimleri = {"Düşük": 1.0, "Orta": 2.0, "Yüksek": 3.0}
    su_secimleri = {"Az": 1.0, "Normal": 2.0, "Fazla": 3.0}
    ogun_secimleri = {"1 öğün": 1.0, "2 öğün": 2.0, "3 öğün": 3.0, "4 öğün": 4.0}
    aktivite_secimleri = {"Hiç yapmıyorum": 0.0, "Düşük": 1.0, "Orta": 2.0, "Yüksek": 3.0}
    teknoloji_secimleri = {"Çok az": 0.0, "Az": 0.5, "Orta": 1.0, "Fazla": 1.5, "Çok fazla": 2.0}

    with st.form("tahmin_formu"):

        col1, col2 = st.columns(2)

        with col1:
            gender_tr = st.selectbox("Cinsiyet", ["Seçiniz"] + list(gender_map.keys()))
            age = st.text_input("Yaş", placeholder="Örn: 21")
            height = st.text_input("Boy (metre)", placeholder="Örn: 1.65")
            weight = st.text_input("Kilo", placeholder="Örn: 65")

            family_history_tr = st.selectbox(
                "Ailede fazla kilo geçmişi var mı?",
                ["Seçiniz"] + list(evet_hayir_map.keys())
            )

            favc_tr = st.selectbox(
                "Yüksek kalorili yiyecek tüketiyor musun?",
                ["Seçiniz"] + list(evet_hayir_map.keys())
            )

            fcvc_tr = st.selectbox(
                "Sebze tüketim sıklığı",
                ["Seçiniz"] + list(sebze_secimleri.keys())
            )

            ncp_tr = st.selectbox(
                "Günlük ana öğün sayısı",
                ["Seçiniz"] + list(ogun_secimleri.keys())
            )

        with col2:
            caec_tr = st.selectbox(
                "Öğün arası yeme alışkanlığı",
                ["Seçiniz"] + list(caec_map.keys())
            )

            smoke_tr = st.selectbox(
                "Sigara kullanıyor musun?",
                ["Seçiniz"] + list(evet_hayir_map.keys())
            )

            ch2o_tr = st.selectbox(
                "Günlük su tüketimi",
                ["Seçiniz"] + list(su_secimleri.keys())
            )

            scc_tr = st.selectbox(
                "Kalori takibi yapıyor musun?",
                ["Seçiniz"] + list(evet_hayir_map.keys())
            )

            faf_tr = st.selectbox(
                "Fiziksel aktivite sıklığı",
                ["Seçiniz"] + list(aktivite_secimleri.keys())
            )

            tue_tr = st.selectbox(
                "Teknoloji kullanım süresi",
                ["Seçiniz"] + list(teknoloji_secimleri.keys())
            )

            calc_tr = st.selectbox(
                "Alkol tüketimi",
                ["Seçiniz"] + list(calc_map.keys())
            )

            mtrans_tr = st.selectbox(
                "Ulaşım türü",
                ["Seçiniz"] + list(mtrans_map.keys())
            )

        tahmin_butonu = st.form_submit_button("Tahmin Et")

    if tahmin_butonu:

        secim_kontrol = [
            gender_tr, family_history_tr, favc_tr, fcvc_tr, ncp_tr,
            caec_tr, smoke_tr, ch2o_tr, scc_tr, faf_tr, tue_tr,
            calc_tr, mtrans_tr
        ]

        if "Seçiniz" in secim_kontrol:
            st.error("Lütfen tüm seçim alanlarını doldurun.")
            st.stop()

        if age.strip() == "" or height.strip() == "" or weight.strip() == "":
            st.error("Lütfen yaş, boy ve kilo alanlarını doldurun.")
            st.stop()

        try:
            age = float(age.replace(",", "."))
            height = float(height.replace(",", "."))
            weight = float(weight.replace(",", "."))

            if age <= 0 or height <= 0 or weight <= 0:
                st.error("Yaş, boy ve kilo değerleri sıfırdan büyük olmalıdır.")
                st.stop()

        except:
            st.error("Lütfen yaş, boy ve kilo alanlarını doğru formatta girin.")
            st.stop()

        bmi = weight / (height ** 2)

        input_data = pd.DataFrame([{
            "Gender": gender_map[gender_tr],
            "Age": age,
            "family_history_with_overweight": evet_hayir_map[family_history_tr],
            "FAVC": evet_hayir_map[favc_tr],
            "FCVC": sebze_secimleri[fcvc_tr],
            "NCP": ogun_secimleri[ncp_tr],
            "CAEC": caec_map[caec_tr],
            "SMOKE": evet_hayir_map[smoke_tr],
            "CH2O": su_secimleri[ch2o_tr],
            "SCC": evet_hayir_map[scc_tr],
            "FAF": aktivite_secimleri[faf_tr],
            "TUE": teknoloji_secimleri[tue_tr],
            "CALC": calc_map[calc_tr],
            "MTRANS": mtrans_map[mtrans_tr],
            "BMI": bmi
        }])

        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_encoded)

        tahmin = model.predict(input_scaled)[0]
        tahmin_orijinal = target_encoder.inverse_transform([tahmin])[0]
        tahmin_tr = sinif_turkce.get(tahmin_orijinal, tahmin_orijinal)

        st.success(f"Tahmin Edilen Obezite Seviyesi: {tahmin_tr}")
        st.info(f"Vücut Kitle İndeksi (BMI): {bmi:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📋 Kişisel Sağlık Önerileri")

        oneriler = []

        if smoke_tr == "Evet":
            oneriler.append("Sigara kullanımı sağlığınız için zararlıdır. Sigara tüketimini bırakmanız veya azaltmanız önerilir.")

        if favc_tr == "Evet":
            oneriler.append("Yüksek kalorili yiyecek tüketimini azaltmanız önerilir. Daha dengeli ve sağlıklı besinler tercih edebilirsiniz.")

        if fcvc_tr == "Düşük":
            oneriler.append("Sebze tüketiminiz düşük görünüyor. Günlük öğünlerinize daha fazla sebze eklemeniz önerilir.")

        if ch2o_tr == "Az":
            oneriler.append("Günlük su tüketiminiz düşük görünüyor. Su tüketiminizi artırmanız sağlığınız için faydalı olabilir.")

        if faf_tr in ["Hiç yapmıyorum", "Düşük"]:
            oneriler.append("Fiziksel aktivite seviyeniz düşük. Haftalık düzenli yürüyüş veya hafif egzersiz yapmanız önerilir.")

        if tue_tr in ["Fazla", "Çok fazla"]:
            oneriler.append("Teknoloji kullanım süreniz yüksek görünüyor. Hareketsiz kalma sürenizi azaltmanız önerilir.")

        if caec_tr in ["Sık sık", "Her zaman"]:
            oneriler.append("Öğün arası yeme alışkanlığınız yüksek görünüyor. Ara öğünlerde daha sağlıklı seçenekler tercih edebilirsiniz.")

        if calc_tr in ["Sık sık", "Her zaman"]:
            oneriler.append("Alkol tüketimini azaltmanız genel sağlığınız açısından faydalı olabilir.")

        if bmi >= 30:
            oneriler.append("BMI değeriniz obezite aralığında görünüyor. Bir uzman desteğiyle beslenme ve egzersiz planı oluşturmanız önerilir.")
        elif bmi >= 25:
            oneriler.append("BMI değeriniz fazla kilo aralığında görünüyor. Dengeli beslenme ve düzenli hareket ile kilo kontrolü sağlanabilir.")
        elif bmi < 18.5:
            oneriler.append("BMI değeriniz düşük görünüyor. Sağlıklı kilo kazanımı için dengeli beslenmeye dikkat etmeniz önerilir.")
        else:
            oneriler.append("BMI değeriniz normal aralıkta görünüyor. Mevcut sağlıklı alışkanlıklarınızı sürdürmeniz önerilir.")

        st.markdown(f"""
        <div style="
            background-color:#f8f8f8;
            border-left:6px solid #800000;
            padding:18px;
            border-radius:12px;
            color:#000000;
            font-size:17px;
            line-height:1.7;
        ">
        <b>Tahmin Edilen Obezite Seviyesi:</b> {tahmin_tr}<br>
        <b>Vücut Kitle İndeksi (BMI):</b> {bmi:.2f}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for i, oneri in enumerate(oneriler, start=1):
            st.markdown(f"""
            <div style="
                background-color:#ffffff;
                border:1px solid #dddddd;
                border-radius:10px;
                padding:12px 16px;
                margin-bottom:10px;
                color:#000000;
                font-size:16px;
            ">
            <b>{i}.</b> {oneri}
            </div>
            """, unsafe_allow_html=True)
        


elif menu == "EDA Analizleri":

    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    st.title("Keşifsel Veri Analizi ve Model Grafikleri")

    df_eda = df_eda_global.copy()

    degisken_turkce = {
        "Gender": "Cinsiyet",
        "Age": "Yaş",
        "Height": "Boy",
        "Weight": "Kilo",
        "family_history_with_overweight": "Ailede Fazla Kilo Geçmişi",
        "FAVC": "Yüksek Kalorili Besin Tüketimi",
        "FCVC": "Sebze Tüketim Sıklığı",
        "NCP": "Ana Öğün Sayısı",
        "CAEC": "Öğün Arası Yeme",
        "SMOKE": "Sigara Kullanımı",
        "CH2O": "Su Tüketimi",
        "SCC": "Kalori Takibi",
        "FAF": "Fiziksel Aktivite",
        "TUE": "Teknoloji Kullanım Süresi",
        "CALC": "Alkol Tüketimi",
        "MTRANS": "Ulaşım Türü",
        "NObeyesdad": "Obezite Seviyesi",
        "BMI": "Vücut Kitle İndeksi"
    }

    deger_turkce = {
        "Female": "Kadın",
        "Male": "Erkek",
        "yes": "Evet",
        "no": "Hayır",
        "Sometimes": "Bazen",
        "Frequently": "Sık sık",
        "Always": "Her zaman",
        "Public_Transportation": "Toplu Taşıma",
        "Walking": "Yürüyüş",
        "Automobile": "Otomobil",
        "Motorbike": "Motosiklet",
        "Bike": "Bisiklet"
    }

    sinif_sirasi_orijinal = [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III"
    ]

    sinif_sirasi_turkce = [
        sinif_turkce.get(s, s)
        for s in sinif_sirasi_orijinal
    ]

    def tr_col(col):
        yeni = str(col)
        for eng, tr in degisken_turkce.items():
            yeni = yeni.replace(eng, tr)
        return yeni

    def tr_val(x):
        return deger_turkce.get(x, x)

    def tr_sinif(x):
        return sinif_turkce.get(x, x)

    if "BMI" not in df_eda.columns:
        if "Height" in df_eda.columns and "Weight" in df_eda.columns:
            df_eda["BMI"] = df_eda["Weight"] / (df_eda["Height"] ** 2)

    if "NObeyesdad" in df_eda.columns:
        df_eda["Obezite_TR"] = df_eda["NObeyesdad"].map(tr_sinif)

    

    sinif_sayilari = (
        df_eda["Obezite_TR"]
        .value_counts()
        .reindex(sinif_sirasi_turkce, fill_value=0)
    )

    fig1, ax1 = plt.subplots(figsize=(5.8, 2.8))

    ax1.bar(
        sinif_sayilari.index,
        sinif_sayilari.values,
        color="#800000"
    )

    ax1.set_xlabel("Obezite Sınıfı", fontsize=7)
    ax1.set_ylabel("Kişi Sayısı", fontsize=7)
    ax1.set_title("Obezite Sınıf Dağılımı", fontsize=9)

    plt.xticks(rotation=20, ha="right", fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()

    st.pyplot(fig1, use_container_width=False)

    

    sayisal_kolonlar = ["BMI", "Age", "FAF", "CH2O", "TUE", "FCVC", "NCP"]
    sayisal_kolonlar = [c for c in sayisal_kolonlar if c in df_eda.columns]

    
    st.header("Korelasyon Matrisi")

    corr_cols = [
    c for c in ["BMI", "Age", "FAF", "CH2O", "TUE", "FCVC", "NCP"]
    if c in df_eda.columns
    ]

    corr = df_eda[corr_cols].corr()
    corr.index = [degisken_turkce.get(x, x) for x in corr.index]
    corr.columns = [degisken_turkce.get(x, x) for x in corr.columns]

    fig_corr, ax_corr = plt.subplots(figsize=(4.8, 3.8))

    im = ax_corr.imshow(
        corr,
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1
    )

    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))

    ax_corr.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=6)
    ax_corr.set_yticklabels(corr.index, fontsize=6)

    ax_corr.set_title(
        "Sayısal Değişkenler Arası Korelasyon",
        fontsize=9,
        fontweight="bold"
    )

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            deger = corr.iloc[i, j]

            ax_corr.text(
                j,
                i,
                f"{deger:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
                color="white" if abs(deger) > 0.45 else "black"
            )

    cbar = fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    st.pyplot(fig_corr, use_container_width=False)
    
    
    from scipy.stats import chi2_contingency
    import numpy as np
    import pandas as pd

    # Kategorik değişkenler
    kategorik_kolonlar = [
        "Gender",
        "family_history_with_overweight",
        "FAVC",
        "CAEC",
        "SMOKE",
        "SCC",
        "CALC",
        "MTRANS"
    ]

    kategorik_kolonlar = [
        c for c in kategorik_kolonlar if c in df_eda.columns
    ]

    # Cramér's V fonksiyonu
    def cramers_v(x, y):

        tablo = pd.crosstab(x, y)

        if tablo.shape[0] < 2 or tablo.shape[1] < 2:
            return 0

        chi2 = chi2_contingency(tablo)[0]

        n = tablo.sum().sum()

        r, k = tablo.shape

        return np.sqrt(chi2 / (n * (min(r, k) - 1)))

    # Cramér's V matrisi oluşturma
    cramer_matrisi = pd.DataFrame(
        index=kategorik_kolonlar,
        columns=kategorik_kolonlar,
        dtype=float
    )

    for col1 in kategorik_kolonlar:
        for col2 in kategorik_kolonlar:
            cramer_matrisi.loc[col1, col2] = cramers_v(
                df_eda[col1],
                df_eda[col2]
            )

    # Türkçe isimler
    cramer_matrisi.index = [
        degisken_turkce.get(x, x)
        for x in cramer_matrisi.index
    ]

    cramer_matrisi.columns = [
        degisken_turkce.get(x, x)
        for x in cramer_matrisi.columns
    ]

    # Grafik
    fig_cramer, ax_cramer = plt.subplots(figsize=(4.8, 4.5))

    im = ax_cramer.imshow(
        cramer_matrisi,
        cmap="YlOrRd",
        vmin=0,
        vmax=1
    )

    ax_cramer.set_xticks(range(len(cramer_matrisi.columns)))
    ax_cramer.set_yticks(range(len(cramer_matrisi.columns)))

    ax_cramer.set_xticklabels(
        cramer_matrisi.columns,
        rotation=35,
        ha="right",
        fontsize=6
    )

    ax_cramer.set_yticklabels(
        cramer_matrisi.index,
        fontsize=6
    )

    ax_cramer.set_title(
        "Kategorik Değişkenler Arası İlişki (Cramér's V)",
        fontsize=9,
        fontweight="bold"
    )

    # Hücre içi değerler
    for i in range(len(cramer_matrisi.index)):
        for j in range(len(cramer_matrisi.columns)):

            deger = cramer_matrisi.iloc[i, j]

            ax_cramer.text(
                j,
                i,
                f"{deger:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
                color="white" if deger > 0.45 else "black"
            )

    # Colorbar
    cbar = fig_cramer.colorbar(
        im,
        ax=ax_cramer,
        fraction=0.046,
        pad=0.04
    )

    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()

    st.pyplot(fig_cramer, use_container_width=False)


    st.header("XGBoost - Özellik Önemi, Karışıklık Matrisi, ROC ve PCA Analizi")

    # Tahminler model eğitim dosyasında kaydedilen y_pred_xgb.pkl dosyasından gelir.
    # Bu nedenle burada tekrar model.predict(X_test_scaled) çalıştırmıyoruz.

    # Sınıf sırası elle değil, doğrudan kaydedilmiş LabelEncoder üzerinden alınmalı.
    # Böylece confusion matrix değerleri terminalde/raporda görünen XGBoost matrisiyle aynı sırada çıkar.
    cm_label_values = target_encoder.transform(target_encoder.classes_)
    sinif_adlari = [
        sinif_turkce.get(s, s)
        for s in target_encoder.classes_
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🔍 Özellik Önemi")

        def ozellik_ana_adi(feature):
            if feature.startswith("Gender_"):
                return "Cinsiyet"
            elif feature.startswith("family_history_with_overweight_"):
                return "Ailede Fazla Kilo Geçmişi"
            elif feature.startswith("FAVC_"):
                return "Yüksek Kalorili Besin Tüketimi"
            elif feature.startswith("CAEC_"):
                return "Öğün Arası Yeme"
            elif feature.startswith("SMOKE_"):
                return "Sigara Kullanımı"
            elif feature.startswith("SCC_"):
                return "Kalori Takibi"
            elif feature.startswith("CALC_"):
                return "Alkol Tüketimi"
            elif feature.startswith("MTRANS_"):
                return "Ulaşım Türü"
            elif feature == "FCVC":
                return "Sebze Tüketim Sıklığı"
            elif feature == "NCP":
                return "Ana Öğün Sayısı"
            elif feature == "CH2O":
                return "Su Tüketimi"
            elif feature == "FAF":
                return "Fiziksel Aktivite"
            elif feature == "TUE":
                return "Teknoloji Kullanım Süresi"
            elif feature == "BMI":
                return "Vücut Kitle İndeksi"
            elif feature == "Age":
                return "Yaş"
            else:
                return feature

        importance_df = pd.DataFrame({
            "Özellik": feature_names,
            "Önem": model.feature_importances_
        })

        importance_df["Özellik"] = importance_df["Özellik"].apply(ozellik_ana_adi)

        importance_df = (
            importance_df
            .groupby("Özellik", as_index=False)["Önem"]
            .sum()
            .sort_values("Önem", ascending=False)
            .head(10)
        )

        fig_imp, ax_imp = plt.subplots(figsize=(5, 4.3))

        ax_imp.barh(
            importance_df["Özellik"],
            importance_df["Önem"],
            color="#800000"
        )

        ax_imp.invert_yaxis()
        ax_imp.set_title("XGBoost Özellik Önemi", fontsize=15, fontweight="bold")
        ax_imp.set_xlabel("Önem Skoru", fontsize=11, fontweight="bold")

        plt.xticks(fontsize=9)
        plt.yticks(fontsize=10, fontweight="bold")
        plt.tight_layout()

        st.pyplot(fig_imp, use_container_width=True)

    

    with col2:
        st.subheader("📌 XGBoost Karışıklık Matrisi")

        cm = confusion_matrix(
            y_test,
            y_pred_xgb,
            labels=cm_label_values
        )

        fig_cm, ax_cm = plt.subplots(figsize=(6.2, 4.3))

        im = ax_cm.imshow(
            cm,
            cmap="Reds"
        )

        ax_cm.set_xticks(range(len(sinif_adlari)))
        ax_cm.set_yticks(range(len(sinif_adlari)))

        ax_cm.set_xticklabels(
            sinif_adlari,
            rotation=35,
            ha="right",
            fontsize=7,
            fontweight="bold"
        )

        ax_cm.set_yticklabels(
            sinif_adlari,
            fontsize=7,
            fontweight="bold"
        )

        ax_cm.set_xlabel("Tahmin", fontsize=10, fontweight="bold")
        ax_cm.set_ylabel("Gerçek", fontsize=10, fontweight="bold")
        ax_cm.set_title("XGBoost Karışıklık Matrisi", fontsize=13, fontweight="bold")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black"
                )

        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test_scaled)

    renkler = {
        sinif_adlari[0]: "#1f77b4",
        sinif_adlari[1]: "#2ca02c",
        sinif_adlari[2]: "#ff7f0e",
        sinif_adlari[3]: "#d62728",
        sinif_adlari[4]: "#9467bd",
        sinif_adlari[5]: "#8c564b",
        sinif_adlari[6]: "#e377c2"
    }

    y_test_orijinal = target_encoder.inverse_transform(y_test)
    y_pred_orijinal = target_encoder.inverse_transform(y_pred_xgb)

    y_test_tr = [sinif_turkce.get(x, x) for x in y_test_orijinal]
    y_pred_tr = [sinif_turkce.get(x, x) for x in y_pred_orijinal]


    with col3:
        st.subheader("🎨 PCA – Gerçek Sınıflar")

        fig_pca_true, ax_pca_true = plt.subplots(figsize=(6.2, 4.3))

        for sinif in sinif_adlari:
            maske = np.array(y_test_tr) == sinif

            ax_pca_true.scatter(
                X_pca[maske, 0],
                X_pca[maske, 1],
                label=sinif,
                s=28,
                alpha=0.75,
                color=renkler[sinif],
                edgecolors="white",
                linewidths=0.3
            )

        ax_pca_true.set_title("PCA Uzayında Gerçek Obezite Sınıfları", fontsize=13, fontweight="bold")
        ax_pca_true.set_xlabel("PC1", fontsize=10, fontweight="bold")
        ax_pca_true.set_ylabel("PC2", fontsize=10, fontweight="bold")
        ax_pca_true.legend(title="Gerçek", fontsize=7, title_fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_pca_true, use_container_width=True)


    with col4:
        st.subheader("🎨 PCA – Tahmin Edilen Sınıflar")

        fig_pca_pred, ax_pca_pred = plt.subplots(figsize=(6.2, 4.3))

        for sinif in sinif_adlari:
            maske = np.array(y_pred_tr) == sinif

            ax_pca_pred.scatter(
                X_pca[maske, 0],
                X_pca[maske, 1],
                label=sinif,
                s=28,
                alpha=0.75,
                color=renkler[sinif],
                edgecolors="white",
                linewidths=0.3
            )

        ax_pca_pred.set_title("PCA Uzayında Tahmin Edilen Obezite Sınıfları", fontsize=13, fontweight="bold")
        ax_pca_pred.set_xlabel("PC1", fontsize=10, fontweight="bold")
        ax_pca_pred.set_ylabel("PC2", fontsize=10, fontweight="bold")
        ax_pca_pred.legend(title="Tahmin", fontsize=7, title_fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_pca_pred, use_container_width=True)
    st.header("ROC Eğrisi Analizi")

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    # ROC için sınıf sırası modelin predict_proba kolon sırasıyla aynı olmalı.
    # XGBoost predict_proba çıktısındaki kolonlar model.classes_ sırasındadır.
    roc_label_values = getattr(model, "classes_", cm_label_values)

    # Gerçek sınıfları binary hale çevir
    y_test_bin = label_binarize(
        y_test,
        classes=roc_label_values
    )

    # Olasılık tahminleri
    y_score = model.predict_proba(X_test_scaled)

    # Türkçe sınıf isimleri - model sınıf sırasına göre oluşturulur
    sinif_adlari = [
        sinif_turkce.get(target_encoder.inverse_transform([int(s)])[0], str(s))
        for s in roc_label_values
    ]

    # Renkler
    renkler = [
        "#800000",
        "#C0392B",
        "#E67E22",
        "#F1C40F",
        "#27AE60",
        "#2980B9",
        "#8E44AD"
    ]

    fig_roc, ax_roc = plt.subplots(figsize=(7, 3))

    for i in range(y_test_bin.shape[1]):

        fpr, tpr, _ = roc_curve(
            y_test_bin[:, i],
            y_score[:, i]
        )

        roc_auc = auc(fpr, tpr)

        ax_roc.plot(
            fpr,
            tpr,
            color=renkler[i],
            linewidth=2,
            label=f"{sinif_adlari[i]} (AUC = {roc_auc:.2f})"
        )

    # Rastgele tahmin çizgisi
    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray"
    )

    ax_roc.set_title(
        "XGBoost ROC Eğrisi",
        fontsize=15,
        fontweight="bold"
    )

    ax_roc.set_xlabel(
        "Yanlış Pozitif Oranı",
        fontsize=11,
        fontweight="bold"
    )

    ax_roc.set_ylabel(
        "Doğru Pozitif Oranı",
        fontsize=11,
        fontweight="bold"
    )

    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig_roc, use_container_width=True)

    st.header("SHAP Özellik Etki Analizi")

    # Kaydedilen SHAP verilerini yükle
    shap_values = joblib.load("shap_values.pkl")
    X_test_shap = joblib.load("X_test_shap.pkl")

    # Özellik isimlerini düzelt
    X_test_shap = pd.DataFrame(
        X_test_shap,
        columns=feature_names
    )

    # Türkçe isimler
    X_test_shap.columns = [
        tr_col(col) for col in X_test_shap.columns
    ]



    fig, ax = plt.subplots(figsize=(8, 5))

    shap.summary_plot(
        shap_values,
        X_test_shap,
        show=False
    )

    st.pyplot(plt.gcf())
    plt.clf()
    