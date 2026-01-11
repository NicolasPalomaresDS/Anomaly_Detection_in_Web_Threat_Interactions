import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Threat Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Cybersecurity Threat Detector")
st.markdown("**Isolation Forest Model**")

# Load pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('pipeline/pipeline.joblib')

pipeline = load_pipeline()

st.markdown("---")

# User inputs
col1, = st.columns(1)

with col1:
    bytes_in = st.number_input("Bytes in", min_value=40, max_value=30000000, value=13182)
    bytes_out = st.number_input("Bytes out", min_value=44, max_value=2000000, value=13799)
    hour = st.slider("Time of day", 0, 23, 8)
    bytes_ratio = bytes_out / (bytes_in + 1)
    st.metric("Bytes ratio", f"{bytes_ratio:.4f}")

# Prediction button
if st.button("🔍 Analyze traffic", type="primary", use_container_width=True):
    
    # Create dataframe
    data = {
        'bytes_in': [bytes_in],
        'bytes_out': [bytes_out],
        'hour': [hour],
        'bytes_ratio': [bytes_ratio]
    }
    
    df = pd.DataFrame(data)
    
    # Prediction
    prediction = pipeline.predict(df)[0]
    score = pipeline.score_samples(df)[0]
    
    # Show results
    st.markdown("---")
    st.subheader("📊 Result")
    
    if prediction == -1:
        st.error("⚠️ **THREAT DETECTED**")
        st.markdown(f"""
        - **Classification**: Anomaly
        - **Anomaly score**: {-score:.4f}
        - **Recomended action**: Investigate the connection
        """)
    else:
        st.success("✅ **NO THREAT DETECTED**")
        st.markdown(f"""
        - **Classification**: Normal
        - **Anomaly score**: {-score:.4f}
        - **State**: No risks detected
        """)
    
    # Detalles adicionales
    with st.expander("🔬 Technical details"):
        st.write("**Input data:**")
        st.json({
            'bytes_in': bytes_in,
            'bytes_out': bytes_out,
            'hour': hour,
            'bytes_ratio': round(bytes_ratio, 4)
        })

st.markdown("---")
st.caption("Model: Isolation Forest | Dataset: CloudWatch Traffic Web Attack")