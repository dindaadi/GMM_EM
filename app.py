import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib.patches import Ellipse

# =============================================
# KONFIGURASI APLIKASI
# =============================================
st.set_page_config(
    page_title="GMM Clustering",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# FUNGSI UTILITAS
# =============================================
def plot_ellipse(gmm, ax):
    """Gambar ellipse berdasarkan mean dan covariance matrix"""
    colors = cm.viridis(np.linspace(0, 1, gmm.n_components))  # Viridis colormap
    for i, (mean, cov, color) in enumerate(zip(gmm.means_, gmm.covariances_, colors)):
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        width, height = 2 * np.sqrt(5.991 * vals)
        ellip = Ellipse(xy=mean, width=width, height=height, angle=angle,
                    edgecolor=color, facecolor='none',fill=False, linestyle='--')
        ax.add_patch(ellip)

def create_silhouette_plot(silhouette_vals, labels, n_components, sil_score):
    """
    Membuat silhouette plot
    
    Parameters:
    -----------
    silhouette_vals : array-like
        Nilai silhouette untuk setiap sample
    labels : array-like
        Label cluster untuk setiap sample
    n_components : int
        Jumlah komponen/cluster
    sil_score : float
        Nilai silhouette score keseluruhan
        
    Returns:
    --------
    fig : matplotlib figure
        Figure yang berisi silhouette plot
    """
    stats_data = []
    # Hitung statistik Silhouette Score per cluster
    for i in range(n_components):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        mean_sil = np.mean(cluster_silhouette_vals)
        min_sil = np.min(cluster_silhouette_vals)
        max_sil = np.max(cluster_silhouette_vals)
        std_sil = np.std(cluster_silhouette_vals)
                    
        stats_data.append({
            "Cluster": i,
            "Mean Silhouette": mean_sil,
            "Min Silhouette": min_sil,
            "Max Silhouette": max_sil,
            "Std Silhouette": std_sil
            })
    # Plot silhouette untuk setiap cluster
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in np.unique(labels):
        cluster_silhouette = silhouette_vals[labels == cluster]
        ax.hist(cluster_silhouette, bins=10, alpha=0.5, label=f'Cluster {cluster}')
    ax.axvline(sil_score, color='red', linestyle='--', label='Avg Silhouette')
    # Pengaturan plot
    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title("Silhouette Plot untuk GMM Clustering", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig, stats_data

def create_cluster_visualization(X_pca, labels, gmm):
    """
    Membuat visualisasi cluster dengan elips dan centroid
    
    Parameters:
    -----------
    X_pca : array-like
        Data yang sudah direduksi dimensinya dengan PCA
    labels : array-like
        Label cluster untuk setiap sampel
    gmm : GaussianMixture
        Model GMM yang sudah dilatih
    
    Returns:
    --------
    fig : matplotlib figure
        Figure yang berisi visualisasi cluster
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot data points
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], 
        c=labels, 
        cmap="viridis", 
        alpha=0.7,
        s=50,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Plot centroids
    ax.scatter(
        gmm.means_[:, 0], gmm.means_[:, 1], 
        marker="X", 
        s=200, 
        cmap="viridis", 
        linewidth=2,
        label="Centroid"
    )
    
    # Tambahkan elips
    plot_ellipse(gmm, ax)
    
    # Tambahkan legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(loc="upper right")
    
    # Pengaturan plot
    ax.set_title("Visualisasi Clustering GMM dengan Batas Elips", fontsize=14)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig

# Fungsi untuk menampilkan hasil numerik clustering GMM
def display_gmm_results(gmm, X_pca):
    st.header("Hasil Numerik Clustering GMM-EM")
    
    # Tampilkan mean setiap cluster
    st.write("Mean setiap cluster:")
    for i, mean in enumerate(gmm.means_):
        st.write(f"Cluster {i}:")
        mean_df = pd.DataFrame(mean)
        st.table(mean_df)

    # Tampilkan proporsi setiap cluster
    st.write("\nProporsi setiap cluster:")
    weight_data = []
    for i, weight in enumerate(gmm.weights_):
        weight_data.append({
            "Cluster": i,
            "Proporsi": weight
        })
    weight_df = pd.DataFrame(weight_data)
    st.table(weight_df)

    # Tampilkan kovarians setiap cluster
    st.write("\nKovarians setiap cluster:")
    for i, cov in enumerate(gmm.covariances_):
        st.write(f"Cluster {i}:")
        cov_df = pd.DataFrame(cov)
        st.table(cov_df)


# =============================================
# APLIKASI UTAMA
# =============================================
def main():
    # Judul sidebar
    st.sidebar.title("🔍 GMM Clustering Explorer")
    st.sidebar.markdown("---")
    
    # Unggah Dataset
    uploaded_file = st.sidebar.file_uploader("📂 Unggah dataset CSV", type=["csv"])
    
    # Menu navigasi yang ditingkatkan dengan tab
    if uploaded_file is not None:
        # Gunakan tabs untuk navigasi yang lebih visual dan modern
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Dataset & Preprocessing", 
            "📊 Visualisasi Clustering", 
            "📈 Evaluasi Silhouette", 
            "🔍 Analisis AIC & BIC"
        ])
        
        # Membaca dataset
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Dataset berhasil diunggah!")
        except Exception as e:
            st.error(f"❌ Error saat membaca file: {e}")
            return
        
        # Sidebar untuk pengaturan
        with st.sidebar:
            st.markdown("## 🛠️ Pengaturan")
            
            # Filter hanya kolom numerik
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_columns) == 0:
                st.error("❌ Dataset tidak memiliki kolom numerik yang diperlukan untuk clustering.")
                return
            
            # Pilih fitur untuk clustering
            selected_columns = st.multiselect(
                "Pilih fitur untuk GMM:",
                options=numeric_columns,
                default=numeric_columns[:min(2, len(numeric_columns))]
            )
            
            # Pilihan standardisasi
            standardization_option = st.radio("Gunakan Standardisasi Data?", ["Tanpa Standardisasi", "Standardisasi (Z-score)", "Min-Max Scaling"])
            
            # Parameter GMM
            st.markdown("### Parameter GMM")
            n_components = st.slider(
                "Jumlah komponen GMM:",
                min_value=2,
                max_value=min(10, len(df) // 20),  # Batas atas berdasarkan ukuran data
                value=3
            )
            
            n_init = st.slider(
                "Jumlah inisialisasi:",
                min_value=1,
                max_value=20,
                value=10,
                help="Jumlah reinisialisasi algoritma dengan berbagai nilai awal"
            )

            max_iter = st.slider(
                "Maksimum Iterasi:", 
                min_value=50, 
                max_value=500, 
                value=100, 
                step=10,
                help="Jumlah iterasi maksimum untuk konvergensi algoritma"
            )

            tolerance = st.sidebar.number_input(
                "Tolerance (default=1e-3):", 
                min_value=0.0, 
                value=1e-3, 
                format="%.3e",
                help="Kriteria konvergensi algoritma"
            )

            # Tombol untuk menjalankan GMM
            run_gmm = st.button("🚀 Jalankan GMM Clustering", type="primary")
        
        # Placeholder untuk hasil GMM
        if "gmm_results" not in st.session_state:
            st.session_state.gmm_results = None
        
        # Jalankan GMM jika tombol ditekan
        if run_gmm:
            with st.spinner("Menjalankan GMM Clustering..."):
                # Ekstrak data dan standarisasi
                X = df[selected_columns].values
                
                if standardization_option == "Standardisasi (Z-score)":
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                elif standardization_option == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X
                    
                # Train GMM
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tolerance,
                    random_state=42
                )
                gmm.fit(X_scaled)
                labels = gmm.predict(X_scaled)

                # Ambil jumlah iterasi hingga konvergensi
                converged_iteration = gmm.n_iter_
                
                # Hitung metrik
                aic = gmm.aic(X_scaled)
                bic = gmm.bic(X_scaled)
                
                # Hitung Silhouette Score jika jumlah cluster > 1
                if n_components > 1:
                    sil_score = silhouette_score(X_scaled, labels)
                    silhouette_vals = silhouette_samples(X_scaled, labels)
                else:
                    sil_score = "N/A"
                    silhouette_vals = None

                # Reduksi dimensi dengan PCA untuk visualisasi
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                gmm_viz = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tolerance,
                    random_state=42
                )
                gmm_viz.fit(X_pca)
                
                # Simpan hasil ke session state
                st.session_state.gmm_results = {
                    "df": df.copy(),
                    "labels": labels,
                    "gmm": gmm,
                    "gmm_viz": gmm_viz,
                    "X_scaled": X_scaled,
                    "X_pca": X_pca,
                    "n_components": n_components,
                    "aic": aic,
                    "bic": bic,
                    "sil_score": sil_score,
                    "silhouette_vals": silhouette_vals,
                    "selected_columns": selected_columns,
                    "converged_iteration": converged_iteration,
                    "MAX_ITER": max_iter
                }
                
                # Tambahkan label ke dataframe
                df["Cluster"] = labels
            
            st.success("✅ GMM Clustering selesai!")
        
        # Tampilkan konten berdasarkan tab
        if st.session_state.gmm_results is not None:
            results = st.session_state.gmm_results
            
            # Tab 1: Dataset & Preprocessing
            with tab1:
                st.header("Dataset & Preprocessing")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Informasi Dataset")
                    st.write(f"**Jumlah baris:** {results['df'].shape[0]}")
                    st.write(f"**Jumlah kolom:** {results['df'].shape[1]}")
                    st.write(f"**Fitur yang digunakan:** {', '.join(results['selected_columns'])}")
                
                with col2:
                    st.subheader("Statistik Deskriptif")
                    st.write(results['df'][results['selected_columns']].describe())
                
                st.subheader("Dataset dengan Label Cluster")
                st.dataframe(results['df'])
                cluster_counts= pd.Series(results['labels']).value_counts().sort_index()

                # Download dataset dengan label
                csv = results['df'].to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Dataset dengan Label Cluster",
                    data=csv,
                    file_name="gmm_clustering_results.csv",
                    mime="text/csv"
                )
            
            # Tab 2: Visualisasi Clustering
            with tab2:
                st.header("Visualisasi Clustering GMM")
                
                st.metric(
                    label=f"Jumlah Iterasi Hingga Konvergensi (Max: {results['MAX_ITER']})",
                    value=f"{results['converged_iteration']}",
                    help="Jumlah iterasi hingga konvergensi algoritma GMM"
                )

                # Visualisasi dengan elips dan centroid
                fig_cluster = create_cluster_visualization(
                    results['X_pca'], 
                    results['labels'], 
                    results['gmm_viz']
                )
                st.pyplot(fig_cluster)
                
                # Visualisasi distribusi probabilitas
                st.subheader("Probabilitas Keanggotaan Sampel")
                
                probs = results['gmm'].predict_proba(results['X_scaled'])
                
                # Sample beberapa data untuk visualisasi
                sample_size = min(100, len(results['X_scaled']))
                sample_indices = np.random.choice(len(results['X_scaled']), sample_size, replace=False)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.imshow(probs[sample_indices], aspect='auto', cmap='viridis')
                ax.set_xlabel("Komponen GMM")
                ax.set_ylabel("Sampel")
                ax.set_title("Probabilitas Keanggotaan untuk 100 Sampel Acak")
                ax.set_xticks(np.arange(results['n_components']))
                ax.set_xticklabels([f"Cluster {i}" for i in range(results['n_components'])])
                plt.colorbar(im, ax=ax, label="Probabilitas")
                
                st.pyplot(fig)
    
                # Tampilkan distribusi cluster sebagai plot
                st.subheader("Distribusi Cluster")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(cluster_counts.index, cluster_counts.values)
                ax.set_xlabel("Cluster")
                ax.set_ylabel("Jumlah Sampel")
                ax.set_title("Distribusi Cluster")
                st.pyplot(fig)

                display_gmm_results(results['gmm'], results['X_pca'])
            
            # Tab 3: Evaluasi Silhouette
            with tab3:
                st.header("Evaluasi Silhouette")
                
                if results['n_components'] <= 1:
                    st.warning("⚠️ Silhouette score hanya dapat dihitung untuk jumlah cluster > 1.")
                else:
                    st.metric(
                        label="Silhouette Score",
                        value=f"{results['sil_score']:.4f}",
                        help="Nilai antara -1 hingga 1. Nilai yang lebih tinggi menunjukkan pemisahan cluster yang lebih baik."
                    )
                    
                    # Silhouette Plot
                    fig_silhouette, stats_data = create_silhouette_plot(
                        results['silhouette_vals'],
                        results['labels'],
                        results['n_components'],
                        results['sil_score']
                    )
                    st.pyplot(fig_silhouette)

                    # Tabel statistik per cluster
                    stats_df = pd.DataFrame(stats_data)
                    st.write("Statistik Silhouette per Cluster:")
                    st.table(stats_df)

                    # Perbandingan Silhouette untuk berbagai jumlah cluster
                    st.subheader("Perbandingan Silhouette Score")
                    
                    with st.spinner("Menghitung Silhouette Score untuk berbagai jumlah cluster..."):
                        sil_scores = []
                        range_n_clusters = range(2, min(8, len(results['df']) // 20) + 1)
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        
                        for i, k in enumerate(range_n_clusters):
                            # Update progress bar
                            progress = (i + 1) / len(range_n_clusters)
                            progress_bar.progress(progress)
                            
                            # Fit GMM
                            gmm_k = GaussianMixture(
                                n_components=k,
                                covariance_type='full',
                                n_init=n_init, 
                                max_iter=max_iter,
                                tol=tolerance, 
                                random_state=42
                                )
                            labels_k = gmm_k.fit_predict(results['X_scaled'])
                            
                            # Hitung Silhouette
                            sil_scores.append(silhouette_score(results['X_scaled'], labels_k))
                        
                        # Reset progress bar
                        progress_bar.empty()
                    
                    # Plot perbandingan Silhouette
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(range_n_clusters, sil_scores, marker='o', linestyle='-', linewidth=2)
                    
                    # Tandai nilai maksimum
                    max_idx = np.argmax(sil_scores)
                    ax.scatter(range_n_clusters[max_idx], sil_scores[max_idx], s=200, c='red', 
                              marker='*', edgecolor='black', zorder=5, 
                              label=f'Max: {range_n_clusters[max_idx]} clusters')
                    
                    ax.set_xlabel("Jumlah Komponen", fontsize=12)
                    ax.set_ylabel("Silhouette Score", fontsize=12)
                    ax.set_title("Silhouette Score untuk Berbagai Jumlah Komponen", fontsize=14)
                    ax.set_xticks(range_n_clusters)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Interpretasi
                    st.info(
                        f"**Interpretasi:** Nilai silhouette score {results['sil_score']:.4f} menunjukkan bahwa cluster "
                        f"{'memiliki separasi yang baik' if results['sil_score'] > 0.5 else 'memiliki separasi yang moderat' if results['sil_score'] > 0.25 else 'memiliki overlap yang signifikan'}. "
                        f"Berdasarkan grafik perbandingan, jumlah komponen optimal adalah **{range_n_clusters[max_idx]}**."
                    )
            
            # Tab 4: Analisis AIC & BIC
            with tab4:
                st.header("Analisis AIC & BIC")
                
                # Tampilkan nilai AIC & BIC saat ini
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="AIC (Akaike Information Criterion)",
                        value=f"{results['aic']:.2f}",
                        help="Nilai yang lebih rendah menunjukkan model yang lebih baik"
                    )
                
                with col2:
                    st.metric(
                        label="BIC (Bayesian Information Criterion)",
                        value=f"{results['bic']:.2f}",
                        help="Nilai yang lebih rendah menunjukkan model yang lebih baik"
                    )
                
                st.subheader("Perbandingan AIC & BIC")
                st.write(
                    "AIC dan BIC adalah metrik untuk memilih jumlah komponen optimal dalam GMM. "
                    "Nilai yang lebih rendah menunjukkan keseimbangan yang lebih baik antara kecocokan model dan kompleksitas."
                )
                
                # Tampilkan grafik perbandingan AIC & BIC
                with st.spinner("Menghitung AIC & BIC untuk berbagai jumlah komponen..."):
                    min_clusters, max_clusters = 2, min(10, len(results['df']) // 20)
                    aic_values, bic_values = [], []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    for i, k in enumerate(range(min_clusters, max_clusters + 1)):
                        # Update progress bar
                        progress = (i + 1) / (max_clusters - min_clusters + 1)
                        progress_bar.progress(progress)
                        
                        # Fit GMM
                        gmm_k = GaussianMixture(
                            n_components=k,
                            covariance_type='full',
                            n_init=n_init,
                            max_iter=max_iter,
                            tol=tolerance,
                            random_state=42
                        )
                        gmm_k.fit(results['X_scaled'])
                        
                        # Hitung AIC & BIC
                        aic_values.append(gmm_k.aic(results['X_scaled']))
                        bic_values.append(gmm_k.bic(results['X_scaled']))
                    
                    # Reset progress bar
                    progress_bar.empty()
                
                # Temukan nilai minimum
                aic_min_idx = np.argmin(aic_values)
                bic_min_idx = np.argmin(bic_values)
                
                # Plot perbandingan AIC & BIC
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(min_clusters, max_clusters + 1), aic_values, marker='o', linestyle='-', 
                      linewidth=2, label="AIC", color="blue")
                ax.plot(range(min_clusters, max_clusters + 1), bic_values, marker='s', linestyle='-', 
                      linewidth=2, label="BIC", color="red")
                
                # Tandai nilai minimum
                ax.scatter(min_clusters + aic_min_idx, aic_values[aic_min_idx], s=200, c='blue', 
                         marker='*', edgecolor='black', zorder=5, 
                         label=f'Min AIC: {min_clusters + aic_min_idx} komponen')
                ax.scatter(min_clusters + bic_min_idx, bic_values[bic_min_idx], s=200, c='red', 
                         marker='*', edgecolor='black', zorder=5, 
                         label=f'Min BIC: {min_clusters + bic_min_idx} komponen')
                
                ax.set_xlabel("Jumlah Komponen", fontsize=12)
                ax.set_ylabel("Nilai Kriteria", fontsize=12)
                ax.set_title("AIC dan BIC untuk Berbagai Jumlah Komponen", fontsize=14)
                ax.set_xticks(range(min_clusters, max_clusters + 1))
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                st.pyplot(fig)
                
                # Interpretasi
                st.info(
                    f"**Interpretasi:** Berdasarkan grafik perbandingan, jumlah komponen optimal adalah "
                    f"**{min_clusters + aic_min_idx}** menurut AIC dan **{min_clusters + bic_min_idx}** menurut BIC. "
                    f"BIC cenderung memilih model yang lebih sederhana dibandingkan AIC."
                )
    else:
        # Tampilan awal sebelum file diunggah
        st.title("🔍 GMM Clustering Explorer")
        st.info("👆 Silakan unggah file CSV melalui panel sidebar untuk memulai.")
        st.markdown(
            """
            ### Cara Menggunakan Aplikasi:
            1. **Unggah Dataset**: Upload file CSV Anda melalui sidebar
            2. **Pilih Fitur**: Pilih kolom yang akan digunakan untuk clustering
            3. **Tentukan Parameter**: Atur jumlah komponen GMM
            4. **Analisis Hasil**: Gunakan tab navigasi untuk melihat hasil
            
            ### Tab yang Tersedia:
            - **Dataset & Preprocessing**: Melihat dataset dan menyiapkan data
            - **Visualisasi Clustering**: Visualisasi hasil clustering dengan PCA dan hasil numerik
            - **Evaluasi Silhouette**: Analisis kualitas clustering dengan silhouette score
            - **Analisis AIC & BIC**: Menentukan jumlah cluster optimal dengan AIC dan BIC
            """
        )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**GMM Clustering Explorer** membantu Anda menganalisis data dengan algoritma Gaussian Mixture Model (GMM) "
        "dan mengevaluasinya menggunakan Silhouette Score, AIC, dan BIC."
    )

if __name__ == "__main__":
    main()
