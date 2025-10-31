import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from io import StringIO
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 🏷️ PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="RiML - Rice Stress Predictor",
    page_icon="🌾",
    layout="wide"
)

# ------------------------------------------------------------
# 🌈 GLOBAL STYLES
# ------------------------------------------------------------
st.markdown("""
    <style>
        .card {
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #2d6a4f !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px !important;
        }
        .stDownloadButton button {
            background-color: #40916c !important;
            color: white !important;
            border-radius: 10px;
            font-weight: 600;
        }
        .stButton button {
            background-color: #2d6a4f !important;
            color: white !important;
            border-radius: 10px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🎛️ TABS
# ------------------------------------------------------------
tab_home, tab_analysis, tab_about = st.tabs(["🏠 Home", "📊 Analysis", "ℹ️ About"])

# ============================================================
# 🏠 HOME TAB
# ============================================================
with tab_home:
    with st.container():
        st.title("🌾 RiML")
        st.markdown("---")

        left_col, right_col = st.columns([2, 1])
        with left_col:
            st.markdown("""
            ### Welcome to **RiML (Rice Machine Learning Stress Predictor)**  
            This tool predicts **rice stress conditions** — *Control, Drought, Salt, or Cold* —  
            using uploaded **gene expression matrices**.

            **Key Capabilities:**
            - 📂 Accepts `.txt`, `.csv`, `.tsv`, `.xlsx` files.  
            - ⚙️ Optional GPL mapping for probe-based datasets.  
            - 🧠 Uses ML model trained on verified rice stress data.  
            - 📊 Produces predictions, confidence levels, and visual summaries.  
            """)
        with right_col:
            st.image(
                "https://th.bing.com/th/id/OIP.yBhgbBkqFi2KS3NMLuan3gHaDt?w=303&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3",
                caption="Rice crop under variable stress conditions",
                use_container_width=True
            )

    with st.expander("💡 How RiML Works"):
        st.markdown("""
        1. Upload gene expression data.  
        2. The app preprocesses, aligns, and scales it to match model genes.  
        3. A trained Random Forest classifier predicts the stress condition.  
        4. Interactive charts visualize results and let you download outputs.
        """)

    with st.expander("🧩 Technologies Used"):
        st.markdown("""
        - **Python** (Pandas, NumPy, Scikit-learn, Streamlit)  
        - **Machine Learning Model:** Random Forest Classifier  
        - **Feature Selection:** Top-ranked stress-related genes  
        """)

    st.markdown("---")
    st.info("👉 Move to the **📊 Analysis** tab to upload your file and get predictions.")

# ============================================================
# 📊 ANALYSIS TAB
# ============================================================
with tab_analysis:
    with st.container():
        st.header("📊 Rice Stress Analysis Dashboard")
        st.markdown("Use this section to upload your dataset and run predictions using the trained RiML model.")
        st.divider()

    # ------------------------------------------------------------
    # 🔧 LOAD MODEL, SCALER, AND FEATURES
    # ------------------------------------------------------------
    @st.cache_resource
    def load_model():
        try:
            model = joblib.load("rice_stress_model.pkl")
            scaler = joblib.load("scaler.pkl")
            try:
                features = list(pd.read_csv("model_features.csv")["Gene"])
                st.toast("✅ Loaded model_features.csv successfully.", icon="✅")
            except FileNotFoundError:
                features = pickle.load(open("top_genes.pkl", "rb"))
            return model, scaler, features
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return None, None, None

    model, scaler, model_genes = load_model()
    if model is None:
        st.stop()

    st.success(f"✅ Model and Scaler loaded. Using {len(model_genes)} genes for analysis.")

    # ------------------------------------------------------------
    # 📁 FILE UPLOAD
    # ------------------------------------------------------------
    with st.container():
        st.subheader("📂 Upload Your Data")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader(
                "Upload expression matrix (.txt, .csv, .tsv, .xlsx)",
                type=["txt", "csv", "tsv", "xlsx"]
            )
        with col2:
            gpl_file = st.file_uploader(
                "Optional: Upload GPL annotation file (.txt, .csv)",
                type=["txt", "csv"]
            )
    st.divider()

    # ------------------------------------------------------------
    # 🔤 HELPER FUNCTIONS
    # ------------------------------------------------------------
    def normalize_name(name):
        return str(name).upper().replace("_", "").replace("-", "").replace(".", "").replace(" ", "")

    def match_genes(user_genes, required_genes):
        required_map = {normalize_name(g): g for g in required_genes}
        return {required_map[n]: u for u in user_genes if (n := normalize_name(u)) in required_map}

    def load_gpl(gpl_file):
        try:
            content = gpl_file.getvalue().decode("utf-8", errors="ignore")
            lines = [l for l in content.split("\n") if not l.startswith("#") and l.strip()]
            processed = "\n".join(lines)
            for sep in ["\t", ","]:
                try:
                    df = pd.read_csv(StringIO(processed), sep=sep)
                    if df.shape[1] >= 2:
                        break
                except Exception:
                    continue
            id_col, gene_col = df.columns[:2]
            mapping = {str(k): str(v).split(" /// ")[0] for k, v in zip(df[id_col], df[gene_col]) if pd.notna(v)}
            st.toast(f"✅ GPL mapping loaded with {len(mapping)} entries.", icon="🧬")
            return mapping
        except Exception as e:
            st.warning(f"⚠️ Could not parse GPL file: {e}")
            return None

    def load_expression(file):
        try:
            if file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                content = file.getvalue().decode("utf-8", errors="ignore")
                lines = [l for l in content.split("\n") if l.strip() and not l.startswith("!")]
                processed = "\n".join(lines)
                for sep in ["\t", ",", ";", r"\s+"]:
                    try:
                        df = pd.read_csv(StringIO(processed), sep=sep, engine="python")
                        if df.shape[1] > 1:
                            break
                    except Exception:
                        continue
            df = df.set_index(df.columns[0]).dropna(how="all")
            return df
        except Exception as e:
            st.error(f"❌ Error loading expression file: {e}")
            return None

    def prepare_data(df, required_genes, gpl_map=None):
        df_mapped = df.copy()
        if gpl_map:
            df_mapped.index = [gpl_map.get(str(i), i) for i in df.index]
        if df_mapped.shape[0] < df_mapped.shape[1]:
            df_mapped = df_mapped.T
        df_mapped = df_mapped.apply(pd.to_numeric, errors="coerce").fillna(0)
        gene_map = match_genes(df_mapped.index, required_genes)
        matched = len(gene_map)
        st.info(f"✅ Matched {matched}/{len(required_genes)} model genes ({100*matched/len(required_genes):.1f}%)")
        if matched == 0:
            st.error("❌ No genes matched! Check your IDs or GPL mapping.")
            return None
        samples = df_mapped.columns
        data_matrix = np.zeros((len(samples), len(required_genes)))
        for i, g in enumerate(required_genes):
            if g in gene_map:
                data_matrix[:, i] = df_mapped.loc[gene_map[g]].values
        return pd.DataFrame(data_matrix, index=samples, columns=required_genes)

    # ------------------------------------------------------------
    # 🚀 MAIN LOGIC
    # ------------------------------------------------------------
    if uploaded_file is not None:
        gpl_map = load_gpl(gpl_file) if gpl_file else None
        expr_df = load_expression(uploaded_file)
        if expr_df is not None:
            st.success(f"✅ Expression data loaded: {expr_df.shape[0]} rows × {expr_df.shape[1]} columns")

            prep_df = prepare_data(expr_df, model_genes, gpl_map)
            if prep_df is not None:
                st.success(f"✨ Prepared {prep_df.shape[0]} samples × {prep_df.shape[1]} features")

                if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
                    with st.spinner("Predicting stress condition..."):
                        try:
                            X_scaled = scaler.transform(prep_df.values)
                            preds = model.predict(X_scaled)
                            probs = model.predict_proba(X_scaled)

                            results = pd.DataFrame({"Sample": prep_df.index, "Predicted_Stress": preds})
                            for i, c in enumerate(model.classes_):
                                results[f"Prob_{c}"] = probs[:, i]
                            results["Confidence"] = probs.max(axis=1)

                            st.subheader("📊 Prediction Results")
                            st.dataframe(results, use_container_width=True, height=400)

                            with st.expander("📈 Visualization"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.bar_chart(results["Predicted_Stress"].value_counts())
                                with col2:
                                    st.line_chart(results["Confidence"])

                            csv = results.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download Predictions (CSV)", csv, "rice_stress_predictions.csv", "text/csv")
                            st.toast("✅ Prediction complete!", icon="🌾")
                        except Exception as e:
                            st.error(f"❌ Prediction error: {e}")
    else:
        st.info("👆 Upload your gene expression matrix to begin analysis.")

# ============================================================
# ℹ️ ABOUT TAB
# ============================================================
with tab_about:
    st.header("ℹ️ About RiML Project")
    st.markdown("---")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 About the Machine Learning Model")
    st.markdown("""
    The **RiML Stress Predictor** uses a supervised learning approach  
    to classify rice samples into **Control**, **Drought**, **Salt**, or **Cold** stress.  
    It leverages **Random Forest** and top gene feature selection to ensure robust accuracy.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Future Trends")
    st.markdown("""
    - Integration with deep learning for multi-stress prediction  
    - Expansion to other crops such as maize, barley, and wheat  
    - Enhanced interpretability with gene importance dashboards  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🙏 Acknowledgement")
    st.markdown("""
    We sincerely thank Dr.Kushagra Kashyap, our guide, for his continuous support, guidance, and encouragement during the development of the RiML – Rice Stress Predictor project.
   His mentorship played a key role in the successful completion of this work.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👩‍🔬 Project Team")
    st.markdown("""
    **Team Name:** RiML Research Group  
    - Pranjal Dalvi (Roll No: 3522411026)  
    - Sejal Chaudari(Roll No: 3522411031) 

    **Institution:**  DES Pune University
    **Year:** 2025
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("© 2025 RiML | Developed with ❤️ using Streamlit 🌾")

