import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import joblib
import pickle
from sklearn.ensemble import ExtraTreesRegressor

# ================== CONFIG ==================
st.set_page_config(page_title="Prediksi Harga Komoditas", layout="wide")

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    raw = {
        'Bawang Merah': pd.read_csv('data/raw/Bawang Merah.csv'),
        'Bawang Putih Bonggol': pd.read_csv('data/raw/Bawang Putih Bonggol.csv'),
        'Beras Medium': pd.read_csv('data/raw/Beras Medium.csv'),
        'Beras Premium': pd.read_csv('data/raw/Beras Premium.csv'),
        'Cabai Merah Keriting': pd.read_csv('data/raw/Cabai Merah Keriting.csv'),
        'Cabai Rawit Merah': pd.read_csv('data/raw/Cabai Rawit Merah.csv'),
        'Daging Ayam Ras': pd.read_csv('data/raw/Daging Ayam Ras.csv'),
        'Daging Sapi Murni': pd.read_csv('data/raw/Daging Sapi Murni.csv'),
        'Gula Konsumsi': pd.read_csv('data/raw/Gula Konsumsi.csv'),
        'Minyak Goreng Curah': pd.read_csv('data/raw/Minyak Goreng Curah.csv'),
        'Minyak Goreng Kemasan Sederhana': pd.read_csv('data/raw/Minyak Goreng Kemasan Sederhana.csv'),
        'Telur Ayam Ras': pd.read_csv('data/raw/Telur Ayam Ras.csv'),
        'Tepung Terigu (Curah)': pd.read_csv('data/raw/Tepung Terigu (Curah).csv'),
    }

    processed_dict = {
        'Bawang Merah': pd.read_csv('data/preprocessed/bawangmerah.csv'),
        'Bawang Putih Bonggol': pd.read_csv('data/preprocessed/bawangputih.csv'),
        'Beras Medium': pd.read_csv('data/preprocessed/berasmedium.csv'),
        'Beras Premium': pd.read_csv('data/preprocessed/beraspremium.csv'),
        'Cabai Merah Keriting': pd.read_csv('data/preprocessed/cabaikeriting.csv'),
        'Cabai Rawit Merah': pd.read_csv('data/preprocessed/cabairawit.csv'),
        'Daging Ayam Ras': pd.read_csv('data/preprocessed/dagingayam.csv'),
        'Daging Sapi Murni': pd.read_csv('data/preprocessed/dagingsapi.csv'),
        'Gula Konsumsi': pd.read_csv('data/preprocessed/gulakonsumsi.csv'),
        'Minyak Goreng Curah': pd.read_csv('data/preprocessed/minyakcurah.csv'),
        'Minyak Goreng Kemasan Sederhana': pd.read_csv('data/preprocessed/minyakkemasan.csv'),
        'Telur Ayam Ras': pd.read_csv('data/preprocessed/telurayam.csv'),
        'Tepung Terigu (Curah)': pd.read_csv('data/preprocessed/tepungterigu.csv'),
    }

    return raw, processed_dict

raw_df, clean_data_dict = load_data()

# ================== INPUT ==================
st.title("📈 Prediksi Harga Komoditas Pangan di Indonesia")
st.markdown("Pilih komoditas dan provinsi untuk melihat tren historis dan prediksi harga.")

komoditas = st.selectbox("Pilih Komoditas", list(clean_data_dict.keys()))
provinsi = st.selectbox("Pilih Provinsi", raw_df[komoditas].columns[1:])
hari = st.slider("Berapa hari ke depan ingin diprediksi?", 1, 360, 30)

# ================== TREN HARGA KOMODITAS ==================
st.subheader("📊 Tren Harga Komoditas (Januari 2022 - September 2024)")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=raw_df[komoditas]['Date'],
    y=raw_df[komoditas][provinsi],
    name="Historis",
    line=dict(color='blue')
))
fig_hist.update_layout(
    title=f"{komoditas.replace('_',' ').title()} - {provinsi}",
    xaxis_title="Tanggal",
    yaxis_title="Harga"
)
st.plotly_chart(fig_hist, use_container_width=True)

# ================== PREDIKSI ==================
if st.button("🔮 Prediksi"):
    with st.spinner("Melakukan prediksi..."):
        df_clean = clean_data_dict[komoditas]
        X = df_clean.drop(df_clean.columns[-1], axis=1)
        y = df_clean[df_clean.columns[-1]].values

        try:
            with open(f"model/{komoditas}.pkl", "rb") as f:
                best_params = pickle.load(f)
        except FileNotFoundError:
            st.error("File parameter terbaik tidak ditemukan untuk komoditas ini.")
            st.stop()

        model = ExtraTreesRegressor(**best_params)
        model.fit(X, y)

        X_input = X.copy()
        predictions = []

        for i in range(hari):
            y_pred = model.predict(X_input)
            y_pred = y_pred.reshape(-1, 1)
            predictions.append(y_pred)
            X_input.iloc[:, :-1] = X_input.iloc[:, 1:].values
            X_input.iloc[:, -1] = y_pred[:, 0]

        hasil_pred = pd.DataFrame(np.hstack(predictions), columns=[f"Hari_{i+1}" for i in range(hari)])

        def inverse_transform(preds, komoditas):
            if komoditas in ['Daging Sapi Murni', 'Tepung Terigu (Curah)','Bawang Putih Bonggol','Gula Konsumsi']:
                return np.square(preds)
            else:
                return np.exp(preds)

        hasil_pred_inv = hasil_pred.apply(lambda row: inverse_transform(row, komoditas), axis=1)

        harga_hist = raw_df[komoditas][provinsi].dropna().values[-30:]
        tanggal_hist = pd.to_datetime(raw_df[komoditas]['Date'].dropna().values[-30:])
        tanggal_akhir = pd.to_datetime(raw_df[komoditas]['Date'].dropna().values[-1])
        tanggal_pred = pd.date_range(start=tanggal_akhir + pd.Timedelta(days=1), periods=hari)

        idx_prov = raw_df[komoditas].columns[1:].tolist().index(provinsi)
        harga_pred = hasil_pred_inv.iloc[idx_prov].values

        st.subheader("📈 Prediksi Harga (Historis + Prediksi)")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=tanggal_hist, y=harga_hist, name="Historis (30 Hari Terakhir)", line=dict(color="blue")))
        fig_pred.add_trace(go.Scatter(x=tanggal_pred, y=harga_pred, name="Prediksi", line=dict(color="red")))
        fig_pred.update_layout(title=f"Prediksi Harga {komoditas.replace('_', ' ').title()} di {provinsi}",
                               xaxis_title="Tanggal", yaxis_title="Harga")
        st.plotly_chart(fig_pred, use_container_width=True)

        st.subheader("📄 Tabel Hasil Prediksi")
        df_pred_show = pd.DataFrame({"Tanggal": tanggal_pred, "Harga": harga_pred})
        st.dataframe(df_pred_show)

        csv = df_pred_show.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Hasil Prediksi", data=csv, file_name=f"prediksi_{komoditas}_{provinsi}.csv", mime='text/csv')


# ================== DASHBOARD ==================
st.markdown("## 🧭 Dashboard Ringkas")

df_raw = raw_df[komoditas][['Date', provinsi]].dropna()
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw = df_raw.sort_values('Date')

harga_terkini = df_raw[provinsi].values[-1]
harga_7hari_lalu = df_raw[provinsi].values[-8] if len(df_raw) >= 8 else harga_terkini
harga_30hari_lalu = df_raw[provinsi].values[-31] if len(df_raw) >= 31 else harga_terkini

perubahan_7hari = harga_terkini - harga_7hari_lalu
persen_7hari = (perubahan_7hari / harga_7hari_lalu * 100) if harga_7hari_lalu != 0 else 0

perubahan_30hari = harga_terkini - harga_30hari_lalu
persen_30hari = (perubahan_30hari / harga_30hari_lalu * 100) if harga_30hari_lalu != 0 else 0

col1, col2, col3 = st.columns(3)

col1.metric("💰 Harga Terakhir", f"{harga_terkini:,.0f}")

col2.metric(
    "📅 Perubahan 7 Hari",
    f"{perubahan_7hari:+,.0f} ({persen_7hari:+.2f}%)",
    delta=f"{persen_7hari:+.2f}%"
)

col3.metric(
    "📆 Perubahan 30 Hari",
    f"{perubahan_30hari:+,.0f} ({persen_30hari:+.2f}%)",
    delta=f"{persen_30hari:+.2f}%"
)

# ================== DASHBOARD BULANAN ==================
st.subheader("📆 Statistik Harga per Bulan")

# Pilihan tahun dan bulan tersedia di data
df_temp = raw_df[komoditas][['Date']].dropna()
df_temp['Date'] = pd.to_datetime(df_temp['Date'])
tahun_tersedia = df_temp['Date'].dt.year.sort_values().unique()
bulan_angka = list(range(1, 13))
bulan_nama = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
              'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
bulan_dict = dict(zip(bulan_nama, bulan_angka))

# Input bulan & tahun
col1, col2 = st.columns(2)
with col1:
    bulan_pilih = st.selectbox("Pilih Bulan", bulan_nama)
with col2:
    tahun_pilih = st.selectbox("Pilih Tahun", tahun_tersedia)

# Filter data
df_bulan = raw_df[komoditas].copy()
df_bulan['Date'] = pd.to_datetime(df_bulan['Date'])
df_bulan['Tahun'] = df_bulan['Date'].dt.year
df_bulan['Bulan'] = df_bulan['Date'].dt.month

data_filtered = df_bulan[
    (df_bulan['Tahun'] == tahun_pilih) &
    (df_bulan['Bulan'] == bulan_dict[bulan_pilih])
]

if data_filtered.empty:
    st.warning("Data tidak tersedia untuk bulan dan tahun yang dipilih.")
else:
    # Hitung rata-rata per provinsi (exclude kolom Date, Tahun, Bulan)
    harga_per_prov = data_filtered.drop(columns=['Date', 'Tahun', 'Bulan']).mean().sort_values(ascending=False)

    # Tampilkan bar chart
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=harga_per_prov.values,
        y=harga_per_prov.index,
        orientation='h',
        marker_color='orange'
    ))
    fig_bar.update_layout(
        title=f"Rata-Rata Harga {komoditas} - {bulan_pilih} {tahun_pilih}",
        xaxis_title="Harga",
        yaxis_title="Provinsi",
        height=600
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Tampilkan tabel
    st.markdown("### 📄 Tabel Harga Rata-Rata per Provinsi")
    df_table = harga_per_prov.reset_index()
    df_table.columns = ['Provinsi', 'Rata-Rata Harga']
    st.dataframe(df_table, use_container_width=True)

