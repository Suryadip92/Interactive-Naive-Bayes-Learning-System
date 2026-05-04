import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Naive Bayes Learning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio > label { color: #94a3b8 !important; font-size: 0.75rem; }

    /* ── Main background ── */
    .stApp { background: #f8fafc; }

    /* ── Step card ── */
    .step-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.08);
        border-left: 4px solid #6366f1;
    }

    /* ── Info / formula boxes ── */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0;
        font-size: 0.9rem;
    }
    .formula-box {
        background: #faf5ff;
        border: 1px solid #ddd6fe;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0;
        font-family: monospace;
        font-size: 0.95rem;
    }
    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0;
    }
    .success-box {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,.08);
    }
    .metric-card h2 { margin: 0; color: #6366f1; }
    .metric-card p  { margin: 0; color: #64748b; font-size: 0.8rem; }

    /* ── Section header ── */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        border-bottom: 2px solid #6366f1;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* ── Pipeline step badge ── */
    .badge {
        display: inline-block;
        background: #6366f1;
        color: white !important;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer { visibility: hidden; }

    hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "df_raw": None, "df_processed": None,
        "target_col": None, "feature_cols": [],
        "encoders": {}, "scaler": None,
        "model": None, "model_type": "Gaussian",
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "y_pred": None, "cv_scores": None,
        "preprocessing_log": [],
        # ── cache keys to avoid re-loading on page switch ──
        "_loaded_source": None,   # "demo:Iris" / "demo:Titanic" / "upload:<filename>"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


MAX_UPLOAD_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# ─────────────────────────────────────────────
# HELPER – detect text/categorical columns
# Covers: object dtype, pandas StringDtype, category
# ─────────────────────────────────────────────
def is_text_col(series):
    """Return True for any column that holds text/categories."""
    dtype_str = str(series.dtype).lower()
    if series.dtype == object:
        return True
    if dtype_str in ("string", "category"):
        return True
    # pandas StringDtype (pd.StringDtype())
    try:
        if isinstance(series.dtype, pd.StringDtype):
            return True
    except Exception:
        pass
    return False

init_state()


# ─────────────────────────────────────────────
# SIDEBAR – NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Naive Bayes")
    st.markdown("### Interactive Learning System")
    st.markdown("---")
    steps = [
        "🏠 Home",
        "📂 1. Dataset Input",
        "🔧 2. Preprocessing",
        "📊 3. EDA",
        "📐 4. NB Theory",
        "⚙️ 5. Training Config",
        "🚀 6. Train & Visualize",
        "🔮 7. Prediction",
        "📈 8. Evaluation",
    ]
    page = st.radio("Navigate", steps, label_visibility="collapsed")
    st.markdown("---")
    if st.session_state.df_raw is not None:
        st.success(f"✅ Dataset loaded\n{st.session_state.df_raw.shape[0]} rows × {st.session_state.df_raw.shape[1]} cols")
    if st.session_state.model is not None:
        st.success("✅ Model trained")

    # ── CLEAR ALL BUTTON ──────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Clear All & Reset", use_container_width=True, type="primary"):
        keys_to_reset = {
            "df_raw": None,
            "df_processed": None,
            "target_col": None,
            "feature_cols": [],
            "encoders": {},
            "scaler": None,
            "model": None,
            "model_type": "Gaussian",
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "y_pred": None,
            "cv_scores": None,
            "preprocessing_log": [],
            "_loaded_source": None,
        }
        for k, v in keys_to_reset.items():
            st.session_state[k] = v
        st.success("All clear")
        st.rerun()

# ─────────────────────────────────────────────
# HELPER – Tooltip pill
# ─────────────────────────────────────────────
def tip(label, text):
    with st.expander(f"💡 Why this step? — *{label}*"):
        st.info(text)


# ═══════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🧠 Naive Bayes Interactive Learning System")
    st.markdown(
        "A step-by-step, visual, and mathematical walkthrough of the "
        "Naive Bayes algorithm — built for engineering students."
    )
    st.markdown("---")

    cols = st.columns(3)
    icons = ["📂", "🔧", "📊", "📐", "⚙️", "🚀", "🔮", "📈"]
    labels = [
        "Dataset Input", "Preprocessing", "EDA",
        "NB Theory", "Training Config", "Train & Visualize",
        "Prediction", "Evaluation"
    ]
    descs = [
        "Upload CSV, preview shape & types",
        "Handle missing values, encode, scale",
        "Distributions, correlations, class balance",
        "Priors, likelihoods, posteriors — step by step",
        "Train/test split & k-fold cross-validation",
        "Learn parameters, visualise distributions",
        "Input a sample, see probability walkthrough",
        "Confusion matrix, accuracy, F1, CV scores",
    ]
    for i, (ic, lb, ds) in enumerate(zip(icons, labels, descs)):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="step-card">
                <span class="badge">Step {i+1}</span>
                <strong>{ic} {lb}</strong><br>
                <small style="color:#64748b">{ds}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>How to use:</strong> Follow the steps in order using the sidebar navigation.
    Upload your CSV dataset first, then proceed through each module.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: DATASET INPUT
# ═══════════════════════════════════════════════════════════════
elif page == "📂 1. Dataset Input":
    st.markdown('<div class="section-header">📂 Dataset Input</div>', unsafe_allow_html=True)

    tip("Dataset Input",
        "Your dataset is the foundation. We need it in CSV format. "
        "A good dataset has a clear target column (what you want to predict) "
        "and several feature columns (what you use to predict).")

    # ── Cached demo loaders (only runs once per demo selection) ──
    @st.cache_data(show_spinner=False)
    def load_iris_demo():
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
        return df

    @st.cache_data(show_spinner=False)
    def load_titanic_demo():
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "Pclass":   np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
            "Sex":      np.random.choice(["male", "female"], n),
            "Age":      np.random.normal(30, 12, n).clip(1, 80).round(1),
            "SibSp":    np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.25, 0.1, 0.05]),
            "Fare":     np.random.exponential(35, n).round(2),
            "Survived": np.random.choice([0, 1], n, p=[0.62, 0.38]),
        })

    @st.cache_data(show_spinner=False)
    def load_uploaded_csv(file_bytes: bytes, filename: str):
        import io
        return pd.read_csv(io.BytesIO(file_bytes))

    # ── File uploader with 1 MB size check ──
    st.markdown(f"#### Upload a CSV file *(max {MAX_UPLOAD_MB} MB)*")
    uploaded = st.file_uploader(
        f"Choose CSV — max {MAX_UPLOAD_MB} MB",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        file_bytes = uploaded.read()
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            st.error(
                f"❌ File too large: **{len(file_bytes)/1024/1024:.2f} MB**. "
                f"Please upload a CSV ≤ {MAX_UPLOAD_MB} MB."
            )
            uploaded = None   # treat as not uploaded
        else:
            source_key = f"upload:{uploaded.name}:{len(file_bytes)}"
            if st.session_state._loaded_source != source_key:
                df_new = load_uploaded_csv(file_bytes, uploaded.name)
                st.session_state.df_raw        = df_new
                st.session_state.df_processed  = df_new.copy()
                st.session_state._loaded_source = source_key
                # reset downstream state when a new dataset is loaded
                for k in ["target_col","feature_cols","encoders","scaler",
                          "model","X_train","X_test","y_train","y_test",
                          "y_pred","cv_scores"]:
                    st.session_state[k] = [] if k == "feature_cols" else ({} if k == "encoders" else None)
            st.success(f"✅ Using uploaded file: **{uploaded.name}** "
                       f"({len(file_bytes)/1024:.1f} KB)")

    # ── Demo datasets ──
    st.markdown("#### Or load a demo dataset")
    demo = st.selectbox("Demo datasets", ["— none —", "Iris (built-in)", "Titanic-lite (built-in)"])

    if demo == "Iris (built-in)":
        source_key = "demo:Iris"
        if st.session_state._loaded_source != source_key:
            df_new = load_iris_demo()
            st.session_state.df_raw        = df_new
            st.session_state.df_processed  = df_new.copy()
            st.session_state._loaded_source = source_key
            for k in ["target_col","feature_cols","encoders","scaler",
                      "model","X_train","X_test","y_train","y_test",
                      "y_pred","cv_scores"]:
                st.session_state[k] = [] if k == "feature_cols" else ({} if k == "encoders" else None)

    elif demo == "Titanic-lite (built-in)":
        source_key = "demo:Titanic"
        if st.session_state._loaded_source != source_key:
            df_new = load_titanic_demo()
            st.session_state.df_raw        = df_new
            st.session_state.df_processed  = df_new.copy()
            st.session_state._loaded_source = source_key
            for k in ["target_col","feature_cols","encoders","scaler",
                      "model","X_train","X_test","y_train","y_test",
                      "y_pred","cv_scores"]:
                st.session_state[k] = [] if k == "feature_cols" else ({} if k == "encoders" else None)

    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.success(f"Dataset loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")

        tab1, tab2, tab3 = st.tabs(["📋 Preview", "🔢 Shape & Types", "❓ Missing Values"])

        with tab1:
            st.dataframe(df.head(20), use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Rows:** {df.shape[0]}  \n**Columns:** {df.shape[1]}")
                st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "dtype"}),
                             use_container_width=True)
            with col2:
                st.dataframe(df.describe(), use_container_width=True)

        with tab3:
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            miss_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
            st.dataframe(miss_df[miss_df["Missing Count"] > 0] if missing.sum() > 0
                         else pd.DataFrame({"Status": ["No missing values ✅"]}),
                         use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎯 Select Target Column")
        target = st.selectbox("Target (class label) column", df.columns.tolist())
        features = [c for c in df.columns if c != target]
        st.session_state.target_col = target
        st.session_state.feature_cols = features
        st.info(f"**Target:** `{target}`  \n**Features ({len(features)}):** {', '.join(features)}")


# ═══════════════════════════════════════════════════════════════
# PAGE: PREPROCESSING
# ═══════════════════════════════════════════════════════════════
elif page == "🔧 2. Preprocessing":
    st.markdown('<div class="section-header">🔧 Preprocessing</div>', unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df_processed.copy()
    target = st.session_state.target_col

    # ── Missing value handling ──────────────────────────────
    st.markdown("### 1️⃣ Missing Value Handling")
    tip("Missing Values",
        "Missing values can skew model training. We replace them (impute) with "
        "a representative value: mean for continuous data (minimises overall error), "
        "mode for categorical data (most frequent class).")

    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0].index.tolist()

    if missing_cols:
        st.warning(f"Columns with missing values: {missing_cols}")
        mv_strategy = st.selectbox("Strategy", ["mean (numeric) / mode (categorical)", "drop rows", "constant (0)"])
        if st.button("Apply Missing Value Fix"):
            before = df.copy()
            for col in missing_cols:
                if df[col].dtype in [np.float64, np.int64]:
                    fill = 0 if mv_strategy == "constant (0)" else df[col].mean()
                    df[col].fillna(fill, inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            if mv_strategy == "drop rows":
                df.dropna(inplace=True)
            st.session_state.df_processed = df
            st.success("✅ Missing values handled")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Before**")
                st.dataframe(before[missing_cols].head())
            with col2:
                st.markdown("**After**")
                st.dataframe(df[missing_cols].head())
    else:
        st.success("✅ No missing values detected")

    st.markdown("---")

    # ── Encoding ────────────────────────────────────────────
    st.markdown("### 2️⃣ Categorical / Text → Numerical Encoding")
    tip("Encoding",
        "Machine learning algorithms work with numbers only. "
        "Any column that contains text or category labels must be converted to integers. "
        "Label Encoding assigns a unique integer to each category (0, 1, 2 …). "
        "One-Hot Encoding creates a new binary column for each category. "
        "'Encode All Columns at Once' is the fastest option — it Label-Encodes every "
        "string/object column including the target in a single click.")

    # Always re-read fresh from session_state so previous steps are reflected
    df = st.session_state.df_processed.copy()
    # Detect ALL string/object columns (features + target)
    all_cat_cols  = [c for c in df.columns if is_text_col(df[c])]
    cat_feat_cols = [c for c in all_cat_cols if c != target]

    if all_cat_cols:
        st.markdown(
            f"**Detected text/categorical columns** "
            f"({len(all_cat_cols)}): `{'`, `'.join(all_cat_cols)}`"
        )

        enc_method = st.selectbox(
            "Encoding method",
            [
                "🔢 Encode ALL Columns at Once (Label Encoding)",
                "Label Encoding — select columns",
                "One-Hot Encoding — select columns",
            ],
        )

        # ── Option A: One-click encode everything ──────────
        if enc_method == "🔢 Encode ALL Columns at Once (Label Encoding)":
            st.markdown("""
            <div class="info-box">
            <strong>What this does:</strong> Every column that contains words/categories
            (including the target column) is converted to integers using Label Encoding.<br>
            Example: <code>['cat', 'dog', 'cat']</code> → <code>[0, 1, 0]</code>
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡ Encode All Text Columns Now"):
                before = df[all_cat_cols].head(5).copy()
                for col in all_cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    st.session_state.encoders[col] = le
                st.session_state.df_processed = df
                st.session_state.feature_cols = [c for c in df.columns if c != target]
                st.success(f"✅ All {len(all_cat_cols)} text column(s) encoded to numbers!")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Before** (text)")
                    st.dataframe(before, use_container_width=True)
                with col2:
                    st.markdown("**After** (numbers)")
                    st.dataframe(df[all_cat_cols].head(5), use_container_width=True)

                # Show mapping for each encoded column
                st.markdown("#### 🗺️ Encoding Maps (what integer = what label)")
                map_cols = st.columns(min(len(all_cat_cols), 4))
                for i, col in enumerate(all_cat_cols):
                    le = st.session_state.encoders[col]
                    mapping = {int(le.transform([cls])[0]): cls for cls in le.classes_}
                    with map_cols[i % len(map_cols)]:
                        st.markdown(f"**`{col}`**")
                        st.dataframe(
                            pd.DataFrame(list(mapping.items()), columns=["Integer", "Original Label"]),
                            use_container_width=True, hide_index=True
                        )

        # ── Option B: Label encode selected columns ─────────
        elif enc_method == "Label Encoding — select columns":
            cols_to_enc = st.multiselect(
                "Choose columns to label-encode", all_cat_cols, default=all_cat_cols
            )
            if cols_to_enc and st.button("Apply Label Encoding"):
                before = df[cols_to_enc].head(4).copy()
                for col in cols_to_enc:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    st.session_state.encoders[col] = le
                st.session_state.df_processed = df
                st.session_state.feature_cols = [c for c in df.columns if c != target]
                st.success("✅ Label Encoding applied")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Before**"); st.dataframe(before)
                with col2:
                    st.markdown("**After**"); st.dataframe(df[cols_to_enc].head(4))

        # ── Option C: One-hot encode selected columns ────────
        else:
            cols_to_enc = st.multiselect(
                "Choose feature columns to one-hot encode",
                cat_feat_cols, default=cat_feat_cols
            )
            if cols_to_enc and st.button("Apply One-Hot Encoding"):
                before = df[cols_to_enc].head(4).copy()
                df = pd.get_dummies(df, columns=cols_to_enc)
                # Still encode target if it's a string
                if is_text_col(df[target]):
                    le_t = LabelEncoder()
                    df[target] = le_t.fit_transform(df[target].astype(str))
                    st.session_state.encoders[target] = le_t
                st.session_state.df_processed = df
                st.session_state.feature_cols = [c for c in df.columns if c != target]
                st.success("✅ One-Hot Encoding applied")
                st.dataframe(df.head(4), use_container_width=True)

    else:
        st.success("✅ No text/categorical columns found — all columns are already numeric!")

    st.markdown("---")

    # ── Feature Scaling ─────────────────────────────────────
    df = st.session_state.df_processed.copy()  # fresh read after encoding step
    st.markdown("### 3️⃣ Feature Scaling")
    tip("Feature Scaling",
        "Naive Bayes (Gaussian) models the distribution of each feature independently, "
        "so scaling doesn't change the prediction mathematically. "
        "However, it helps with numerical stability and is good practice.")

    scale_method = st.selectbox("Scaling method", ["None", "StandardScaler (Z-score)", "MinMaxScaler (0-1)"])
    num_cols = [c for c in st.session_state.feature_cols if c in df.columns and df[c].dtype in [np.float64, np.int64]]

    if scale_method != "None":
        if st.button("Apply Scaling"):
            before = df[num_cols].head(3).copy()
            if scale_method == "StandardScaler (Z-score)":
                sc = StandardScaler()
            else:
                sc = MinMaxScaler()
            df[num_cols] = sc.fit_transform(df[num_cols])
            st.session_state.scaler = sc
            st.session_state.df_processed = df
            st.success("✅ Scaling applied")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Before**"); st.dataframe(before)
            with col2:
                st.markdown("**After**"); st.dataframe(df[num_cols].head(3))
    else:
        st.info("No scaling applied.")

    st.markdown("---")
    st.markdown("### ✅ Preprocessed Dataset Preview")
    st.dataframe(st.session_state.df_processed.head(), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════
elif page == "📊 3. EDA":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if st.session_state.df_processed is None:
        st.warning("⚠️ Please complete dataset input first."); st.stop()

    df = st.session_state.df_processed
    target = st.session_state.target_col
    features = [c for c in st.session_state.feature_cols if c in df.columns]

    # ── Class distribution ──────────────────────────────────
    st.markdown("### 🎯 Class Distribution")
    tip("Class Distribution",
        "If one class massively outnumbers others (class imbalance), "
        "the model may bias toward the majority class. "
        "Naive Bayes handles this via prior probabilities, but it's good to be aware.")

    class_counts = df[target].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        class_counts.plot(kind="bar", ax=ax, color="#6366f1", edgecolor="white", width=0.6)
        ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Class"); ax.set_ylabel("Count")
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        plt.tight_layout(); st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(class_counts, labels=class_counts.index.astype(str), autopct="%1.1f%%",
               colors=["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#3b82f6"])
        ax.set_title("Class Proportion", fontsize=13, fontweight="bold")
        fig.patch.set_facecolor("#f8fafc")
        plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")

    # ── Feature distributions ────────────────────────────────
    st.markdown("### 📈 Feature Distributions")
    tip("Feature Distributions",
        "Histograms reveal whether features follow a Gaussian (bell-shaped) distribution. "
        "Gaussian Naive Bayes assumes this — so checking it matters!")

    num_features = [f for f in features if df[f].dtype in [np.float64, np.int64]]
    if num_features:
        cols_per_row = 3
        for i in range(0, len(num_features), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, feat in enumerate(num_features[i:i+cols_per_row]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(4, 2.8))
                    for cls in df[target].unique():
                        subset = df[df[target] == cls][feat].dropna()
                        ax.hist(subset, bins=20, alpha=0.6, label=str(cls), edgecolor="none")
                    ax.set_title(feat, fontweight="bold")
                    ax.legend(fontsize=7)
                    ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
                    plt.tight_layout(); st.pyplot(fig)
    else:
        st.info("No numeric features to plot.")

    st.markdown("---")

    # ── Correlation heatmap ──────────────────────────────────
    st.markdown("### 🔥 Correlation Analysis")
    tip("Correlation",
        "High correlation between features violates the 'naïve' independence assumption. "
        "Naive Bayes still works in practice, but be aware of correlated features.")

    # Build a numeric-only copy — label-encode any remaining string/category columns
    all_cols_for_corr = [c for c in (features + ([target] if target else [])) if c in df.columns]
    corr_df = df[all_cols_for_corr].copy()

    encoded_cols = []
    for col in corr_df.columns.tolist():
        if is_text_col(corr_df[col]):
            corr_df[col] = LabelEncoder().fit_transform(corr_df[col].astype(str))
            encoded_cols.append(col)

    # Drop anything that still can't be cast to float
    for col in corr_df.columns.tolist():
        try:
            corr_df[col] = corr_df[col].astype(float)
        except Exception:
            corr_df.drop(columns=[col], inplace=True)

    if encoded_cols:
        st.info(
            f"ℹ️ Categorical columns were **label-encoded for correlation only** "
            f"(original dataset unchanged): `{'`, `'.join(encoded_cols)}`"
        )

    if len(corr_df.columns) > 1:
        n = len(corr_df.columns)
        fig, ax = plt.subplots(figsize=(max(6, n), max(4, n - 1)))
        corr = corr_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=-1, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title("Feature Correlation Matrix (categorical → label encoded)", fontsize=13, fontweight="bold")
        fig.patch.set_facecolor("#f8fafc")
        plt.tight_layout(); st.pyplot(fig)
    else:
        st.info("Not enough columns to compute a correlation matrix.")


# ═══════════════════════════════════════════════════════════════
# PAGE: NB THEORY
# ═══════════════════════════════════════════════════════════════
elif page == "📐 4. NB Theory":
    st.markdown('<div class="section-header">📐 Naive Bayes Theory</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Naive Bayes is a probabilistic classifier based on <strong>Bayes' Theorem</strong> with the
    "naïve" assumption that all features are conditionally independent given the class.
    </div>
    """, unsafe_allow_html=True)

    # ── Bayes' Theorem ──────────────────────────────────────
    st.markdown("### 1️⃣ Bayes' Theorem")
    st.latex(r"P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="formula-box"><b>P(C|X)</b><br>Posterior<br><small>Prob of class C given features X</small></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="formula-box"><b>P(X|C)</b><br>Likelihood<br><small>Prob of features X given class C</small></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="formula-box"><b>P(C)</b><br>Prior<br><small>Overall prob of class C</small></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="formula-box"><b>P(X)</b><br>Evidence<br><small>Normalisation constant (same for all classes)</small></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Naive independence assumption ────────────────────────
    st.markdown("### 2️⃣ Conditional Independence Assumption")
    st.markdown('<div class="info-box">Each feature <em>x<sub>i</sub></em> is assumed to be independent of every other feature given the class label C. This simplifies the joint likelihood.</div>', unsafe_allow_html=True)
    st.latex(r"P(X \mid C) = \prod_{i=1}^{n} P(x_i \mid C)")

    st.markdown("---")

    # ── Prior probability ────────────────────────────────────
    st.markdown("### 3️⃣ Prior Probability")
    st.latex(r"P(C = c) = \frac{\text{Number of samples with class } c}{\text{Total samples}}")

    if st.session_state.df_processed is not None and st.session_state.target_col:
        df = st.session_state.df_processed
        target = st.session_state.target_col
        counts = df[target].value_counts()
        total = len(df)
        priors = (counts / total).round(4)
        prior_df = pd.DataFrame({"Class": priors.index, "Count": counts.values, "Prior P(C)": priors.values})
        st.dataframe(prior_df, use_container_width=True)
    else:
        st.info("Load a dataset to see computed priors here.")

    st.markdown("---")

    # ── Gaussian NB Likelihood ───────────────────────────────
    st.markdown("### 4️⃣ Gaussian Naive Bayes — Likelihood")
    st.markdown('<div class="info-box">For continuous features, Gaussian NB models each feature as a Gaussian (Normal) distribution per class.</div>', unsafe_allow_html=True)
    st.latex(r"P(x_i \mid C) = \frac{1}{\sqrt{2\pi\sigma_{iC}^2}} \exp\!\left(-\frac{(x_i - \mu_{iC})^2}{2\sigma_{iC}^2}\right)")
    st.markdown("**Parameters learned:** mean (μ) and variance (σ²) per feature per class.")

    st.markdown("---")

    # ── Multinomial NB ──────────────────────────────────────
    st.markdown("### 5️⃣ Multinomial Naive Bayes — Likelihood")
    st.markdown('<div class="info-box">For discrete/count features (e.g., word counts), Multinomial NB estimates the probability of each feature value per class with Laplace smoothing.</div>', unsafe_allow_html=True)
    st.latex(r"P(x_i \mid C) = \frac{\text{count}(x_i, C) + \alpha}{\sum_j \text{count}(x_j, C) + \alpha \cdot n}")

    st.markdown("---")

    # ── Decision Rule ────────────────────────────────────────
    st.markdown("### 6️⃣ Decision Rule (MAP)")
    st.latex(r"\hat{C} = \underset{c}{\arg\max} \; P(C=c) \prod_{i=1}^{n} P(x_i \mid C=c)")

    st.markdown("---")

    # ── Step-by-step toy example ─────────────────────────────
    st.markdown("### 🎲 Step-by-Step Worked Example")
    st.markdown("**Toy dataset:** Weather → Play Tennis?")

    toy = pd.DataFrame({
        "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
        "Temp":    ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
        "Play":    ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
    })
    st.dataframe(toy, use_container_width=True)

    st.markdown("**Predict Play for: Outlook=Sunny, Temp=Cool**")

    yes = toy[toy.Play == "Yes"]
    no  = toy[toy.Play == "No"]

    p_yes = len(yes)/len(toy); p_no = len(no)/len(toy)
    p_sunny_yes  = len(yes[yes.Outlook=="Sunny"])/len(yes)
    p_cool_yes   = len(yes[yes.Temp=="Cool"])/len(yes)
    p_sunny_no   = len(no[no.Outlook=="Sunny"])/len(no)
    p_cool_no    = len(no[no.Temp=="Cool"])/len(no)

    post_yes = p_yes * p_sunny_yes * p_cool_yes
    post_no  = p_no  * p_sunny_no  * p_cool_no

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="formula-box">
        <b>P(Yes) = {p_yes:.3f}</b><br>
        P(Sunny | Yes) = {p_sunny_yes:.3f}<br>
        P(Cool  | Yes) = {p_cool_yes:.3f}<br><br>
        Numerator = {p_yes:.3f} × {p_sunny_yes:.3f} × {p_cool_yes:.3f}<br>
        = <strong>{post_yes:.4f}</strong>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="formula-box">
        <b>P(No) = {p_no:.3f}</b><br>
        P(Sunny | No) = {p_sunny_no:.3f}<br>
        P(Cool  | No) = {p_cool_no:.3f}<br><br>
        Numerator = {p_no:.3f} × {p_sunny_no:.3f} × {p_cool_no:.3f}<br>
        = <strong>{post_no:.4f}</strong>
        </div>
        """, unsafe_allow_html=True)

    winner = "Yes ✅" if post_yes > post_no else "No ❌"
    st.markdown(f"""
    <div class="success-box">
    <strong>Prediction: Play = {winner}</strong> (larger unnormalised posterior wins)
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: TRAINING CONFIG
# ═══════════════════════════════════════════════════════════════
elif page == "⚙️ 5. Training Config":
    st.markdown('<div class="section-header">⚙️ Training Configuration</div>', unsafe_allow_html=True)

    if st.session_state.df_processed is None:
        st.warning("⚠️ Please complete preprocessing first."); st.stop()

    df  = st.session_state.df_processed
    target = st.session_state.target_col
    features = [c for c in st.session_state.feature_cols if c in df.columns]

    st.markdown("### 🔀 Train / Test Split")
    tip("Train/Test Split",
        "We train on one portion and evaluate on the unseen remainder. "
        "A typical split is 80% train / 20% test. "
        "The test set simulates real-world, unseen data.")

    split_ratio = st.slider("Test set size (%)", 10, 40, 20) / 100
    random_state = st.number_input("Random seed", value=42, step=1)

    st.markdown("### 🤖 Model Type")
    tip("Model Type",
        "Gaussian NB: best for continuous, normally distributed features. "
        "Multinomial NB: best for discrete/count features (must be non-negative).")
    model_type = st.radio("Naive Bayes variant", ["Gaussian NB", "Multinomial NB"])

    st.markdown("### 🔁 Cross-Validation")
    tip("Cross-Validation",
        "k-fold CV splits data into k equal folds, trains on k-1, evaluates on 1, "
        "and rotates k times. Average score is more reliable than a single split.")
    use_cv = st.checkbox("Enable k-fold Cross-Validation")
    k = 5
    if use_cv:
        k = st.slider("Number of folds (k)", 2, 10, 5)

    if st.button("✅ Apply Configuration & Split Data"):

        # ── Always use the latest processed dataframe ──────────
        df_fresh = st.session_state.df_processed.copy()
        features_fresh = [c for c in st.session_state.feature_cols if c in df_fresh.columns]

        # ── Auto-encode every remaining string/category column ──
        X_df = df_fresh[features_fresh].copy()
        auto_encoded = []
        for col in X_df.columns:
            if is_text_col(X_df[col]):
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                auto_encoded.append(col)

        # ── Encode target if still string ───────────────────────
        y_series = df_fresh[target].copy()
        if is_text_col(y_series):
            le_y = LabelEncoder()
            y_series = pd.Series(
                le_y.fit_transform(y_series.astype(str)),
                index=y_series.index
            )
            if target not in auto_encoded:
                auto_encoded.append(target)

        if auto_encoded:
            st.warning(
                f"⚠️ These columns were still text — **auto Label-Encoded** for training: "
                f"`{'`, `'.join(auto_encoded)}`\n\n"
                f"👉 Go to **Step 2 → Preprocessing → Encode ALL Columns** to do this properly before training."
            )

        # ── Safe conversion: force each column to float ─────────
        for col in X_df.columns:
            try:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0).astype(float)
            except Exception:
                X_df[col] = 0.0

        X = X_df.values.astype(float)
        y = y_series.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=int(random_state),
            stratify=y if len(np.unique(y)) > 1 else None
        )

        st.session_state.X_train    = X_train
        st.session_state.X_test     = X_test
        st.session_state.y_train    = y_train
        st.session_state.y_test     = y_test
        st.session_state.model_type = "Gaussian" if "Gaussian" in model_type else "Multinomial"

        if use_cv:
            if "Gaussian" in model_type:
                m_tmp = GaussianNB()
                X_cv  = X
            else:
                X_cv  = np.abs(X).astype(int)
                m_tmp = MultinomialNB()
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            scores = cross_val_score(m_tmp, X_cv, y, cv=cv, scoring="accuracy")
            st.session_state.cv_scores = scores
            st.success(f"✅ CV done — Mean Accuracy: **{scores.mean():.4f}** ± {scores.std():.4f}")
            st.bar_chart(
                pd.DataFrame({"Fold Accuracy": scores,
                              "Fold": [f"Fold {i+1}" for i in range(k)]}).set_index("Fold")
            )
        else:
            st.session_state.cv_scores = None

        st.success(f"✅ Data split — Train: {len(y_train)}, Test: {len(y_test)}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training samples", len(y_train))
        with col2:
            st.metric("Test samples", len(y_test))


# ═══════════════════════════════════════════════════════════════
# PAGE: TRAIN & VISUALIZE
# ═══════════════════════════════════════════════════════════════
elif page == "🚀 6. Train & Visualize":
    st.markdown('<div class="section-header">🚀 Train & Visualize</div>', unsafe_allow_html=True)

    if st.session_state.X_train is None:
        st.warning("⚠️ Please complete Training Configuration first."); st.stop()

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    features = [c for c in st.session_state.feature_cols if c in st.session_state.df_processed.columns]

    if st.button("🚀 Train Model"):
        if st.session_state.model_type == "Gaussian":
            model = GaussianNB()
        else:
            X_train = np.abs(X_train).astype(int)
            model = MultinomialNB()

        model.fit(X_train, y_train)
        st.session_state.model = model

        # Run prediction on test set
        X_test = st.session_state.X_test
        if st.session_state.model_type == "Multinomial":
            X_test = np.abs(X_test).astype(int)
            st.session_state.X_test = X_test
        st.session_state.y_pred = model.predict(X_test)
        st.success(f"✅ {st.session_state.model_type} Naive Bayes trained!")

    if st.session_state.model is not None:
        model = st.session_state.model
        classes = model.classes_

        st.markdown("### 🎯 Class Prior Probabilities")
        # class_prior_ exists for GaussianNB; MultinomialNB uses class_log_prior_
        if hasattr(model, "class_prior_"):
            priors = model.class_prior_
        else:
            priors = np.exp(model.class_log_prior_)
        prior_df = pd.DataFrame({"Class": classes, "Prior P(C)": priors.round(4)})
        st.dataframe(prior_df, use_container_width=True)

        # Show Multinomial feature log-probabilities
        if st.session_state.model_type == "Multinomial" and hasattr(model, "feature_log_prob_"):
            st.markdown("### 📐 Learned Parameters (Multinomial NB)")
            tip("Multinomial Parameters",
                "Multinomial NB stores the log-probability of each feature given each class. "
                "Higher value = that feature appears more often in that class.")
            feat_prob_df = pd.DataFrame(
                np.exp(model.feature_log_prob_),
                columns=features,
                index=[f"Class {c}" for c in classes]
            )
            st.markdown("#### Feature Probabilities P(feature | class)")
            st.dataframe(feat_prob_df.round(4), use_container_width=True)

        if st.session_state.model_type == "Gaussian" and hasattr(model, "theta_"):
            st.markdown("### 📐 Learned Parameters (Gaussian NB)")
            tip("Learned Parameters",
                "For each class, Gaussian NB stores the mean (μ) and variance (σ²) "
                "of each feature. These define the Gaussian distribution used to compute P(x|C).")

            st.markdown("#### Mean (μ) per class per feature")
            means_df = pd.DataFrame(model.theta_, columns=features, index=[f"Class {c}" for c in classes])
            st.dataframe(means_df.round(4), use_container_width=True)

            st.markdown("#### Variance (σ²) per class per feature")
            vars_df = pd.DataFrame(model.var_, columns=features, index=[f"Class {c}" for c in classes])
            st.dataframe(vars_df.round(4), use_container_width=True)

            # ── Gaussian curves ──────────────────────────────
            st.markdown("### 📈 Gaussian Likelihood Curves")
            selected_feat = st.selectbox("Select feature to visualise", features)
            feat_idx = features.index(selected_feat)

            fig, ax = plt.subplots(figsize=(8, 4))
            x_range = np.linspace(
                model.theta_[:, feat_idx].min() - 3 * np.sqrt(model.var_[:, feat_idx].max()),
                model.theta_[:, feat_idx].max() + 3 * np.sqrt(model.var_[:, feat_idx].max()),
                300
            )
            colors = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#3b82f6"]
            for idx, cls in enumerate(classes):
                mu = model.theta_[idx, feat_idx]
                sigma = np.sqrt(model.var_[idx, feat_idx])
                y_gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
                ax.plot(x_range, y_gauss, label=f"Class {cls} (μ={mu:.2f}, σ={sigma:.2f})",
                        color=colors[idx % len(colors)], linewidth=2.5)
                ax.fill_between(x_range, y_gauss, alpha=0.12, color=colors[idx % len(colors)])

            ax.set_title(f"Gaussian Likelihood — {selected_feat}", fontsize=13, fontweight="bold")
            ax.set_xlabel(selected_feat); ax.set_ylabel("P(x | C)")
            ax.legend(); ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
            plt.tight_layout(); st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ═══════════════════════════════════════════════════════════════
elif page == "🔮 7. Prediction":
    st.markdown('<div class="section-header">🔮 Prediction & Inference</div>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first."); st.stop()

    model = st.session_state.model
    features = [c for c in st.session_state.feature_cols if c in st.session_state.df_processed.columns]
    df = st.session_state.df_processed

    st.markdown("### 🎛️ Enter a New Sample")
    tip("Prediction",
        "Enter feature values for an unseen sample. "
        "The system will compute the posterior probability for each class "
        "and return the class with the highest probability.")

    input_vals = {}
    cols = st.columns(min(len(features), 4))
    for i, feat in enumerate(features):
        with cols[i % len(cols)]:
            col_min = float(df[feat].min()) if df[feat].dtype in [np.float64, np.int64] else 0.0
            col_max = float(df[feat].max()) if df[feat].dtype in [np.float64, np.int64] else 1.0
            col_mean = float(df[feat].mean()) if df[feat].dtype in [np.float64, np.int64] else 0.0
            input_vals[feat] = st.number_input(feat, value=round(col_mean, 3),
                                               min_value=col_min - abs(col_min),
                                               max_value=col_max + abs(col_max),
                                               step=(col_max - col_min) / 100 if col_max != col_min else 0.01)

    if st.button("🔮 Predict"):
        X_new = np.array([[input_vals[f] for f in features]])
        if st.session_state.model_type == "Multinomial":
            X_new = np.abs(X_new).astype(int)

        pred_class = model.predict(X_new)[0]
        pred_proba = model.predict_proba(X_new)[0]
        classes = model.classes_

        st.markdown("### 📊 Step-by-Step Probability Computation")
        log_priors = model.class_log_prior_ if hasattr(model, "class_log_prior_") else np.log(model.class_prior_)

        if st.session_state.model_type == "Gaussian":
            log_likelihoods = np.zeros(len(classes))
            for ci, cls in enumerate(classes):
                for fi, feat in enumerate(features):
                    mu = model.theta_[ci, fi]
                    sigma2 = model.var_[ci, fi]
                    xi = X_new[0, fi]
                    ll = -0.5 * np.log(2 * np.pi * sigma2) - (xi - mu)**2 / (2 * sigma2)
                    log_likelihoods[ci] += ll

            detail_rows = []
            for ci, cls in enumerate(classes):
                detail_rows.append({
                    "Class": str(cls),
                    "log P(C)": round(log_priors[ci], 4),
                    "Σ log P(X|C)": round(log_likelihoods[ci], 4),
                    "log Posterior": round(log_priors[ci] + log_likelihoods[ci], 4),
                    "P(C|X)": round(pred_proba[ci], 4),
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

        # Bar chart of posteriors
        fig, ax = plt.subplots(figsize=(6, 3))
        bar_colors = ["#6366f1" if c == pred_class else "#cbd5e1" for c in classes]
        ax.barh([str(c) for c in classes], pred_proba, color=bar_colors, edgecolor="white")
        ax.set_xlabel("Posterior Probability")
        ax.set_title("Posterior Probabilities P(C|X)", fontweight="bold")
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        plt.tight_layout(); st.pyplot(fig)

        st.markdown(f"""
        <div class="success-box" style="font-size:1.1rem">
        <strong>🎯 Predicted Class: {pred_class}</strong>
        &nbsp;&nbsp;(confidence: {max(pred_proba)*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ═══════════════════════════════════════════════════════════════
elif page == "📈 8. Evaluation":
    st.markdown('<div class="section-header">📈 Evaluation Metrics</div>', unsafe_allow_html=True)

    if st.session_state.y_pred is None:
        st.warning("⚠️ Please train the model first."); st.stop()

    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    classes = st.session_state.model.classes_
    avg = "binary" if len(classes) == 2 else "macro"

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)

    # ── Metric cards ─────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    for col, name, val, color in [
        (col1, "Accuracy",  acc,  "#6366f1"),
        (col2, "Precision", prec, "#10b981"),
        (col3, "Recall",    rec,  "#f59e0b"),
        (col4, "F1-Score",  f1,   "#ef4444"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color:{color}">{val:.4f}</h2>
                <p>{name}</p>
            </div>
            """, unsafe_allow_html=True)

    tip("Evaluation Metrics",
        "Accuracy = correct predictions / total. "
        "Precision = true positives / (true+false positives) — how precise is 'positive' prediction. "
        "Recall = true positives / (true positives+false negatives) — did we catch all positives? "
        "F1 = harmonic mean of Precision and Recall.")

    st.markdown("---")

    # ── Confusion Matrix ─────────────────────────────────────
    st.markdown("### 🟦 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[str(c) for c in classes],
                yticklabels=[str(c) for c in classes],
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#f8fafc")
    plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")

    # ── Classification Report ────────────────────────────────
    st.markdown("### 📋 Full Classification Report")
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], output_dict=True)
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)

    # ── CV scores ────────────────────────────────────────────
    if st.session_state.cv_scores is not None:
        st.markdown("---")
        st.markdown("### 🔁 Cross-Validation Results")
        cv_scores = st.session_state.cv_scores
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
            "Accuracy": cv_scores
        })
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(cv_df.Fold, cv_df.Accuracy, color="#6366f1", edgecolor="white", width=0.6)
            ax.axhline(cv_scores.mean(), color="#ef4444", linestyle="--", label=f"Mean: {cv_scores.mean():.4f}")
            ax.set_ylim(0, 1.05)
            ax.set_title("Cross-Validation Accuracy per Fold", fontweight="bold")
            ax.legend()
            ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
            plt.tight_layout(); st.pyplot(fig)
        with col2:
            st.metric("Mean CV Accuracy", f"{cv_scores.mean():.4f}")
            st.metric("Std Dev", f"{cv_scores.std():.4f}")
            st.metric("Min Fold", f"{cv_scores.min():.4f}")
            st.metric("Max Fold", f"{cv_scores.max():.4f}")