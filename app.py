import streamlit as st
from src.inference import get_prediction

# Initialise session state variable
if 'input_features' not in st.session_state:
    st.session_state['input_features'] = {}

FIELD_CONFIG = {
    "mean_radius": {"label": "Mean Radius", "min": 7.0, "max": 28.2, "value": 14.1, "step": 0.1},
    "mean_texture": {"label": "Mean Texture", "min": 9.5, "max": 39.5, "value": 19.3, "step": 0.1},
    "mean_perimeter": {"label": "Mean Perimeter", "min": 43.0, "max": 189.0, "value": 92.0, "step": 0.5},
    "mean_area": {"label": "Mean Area", "min": 140.0, "max": 2505.0, "value": 655.0, "step": 1.0},
    "mean_smoothness": {"label": "Mean Smoothness", "min": 0.05, "max": 0.17, "value": 0.10, "step": 0.001},
    "mean_compactness": {"label": "Mean Compactness", "min": 0.02, "max": 0.35, "value": 0.10, "step": 0.001},
    "mean_concavity": {"label": "Mean Concavity", "min": 0.0, "max": 0.45, "value": 0.09, "step": 0.001},
}


def app_sidebar():
    st.sidebar.header("Cell Features")
    st.sidebar.caption("Adjust the seven tumour morphology features and assess the screening risk.")

    def get_input_features():
        input_features = {}
        for field_name, field_meta in FIELD_CONFIG.items():
            input_features[field_name] = st.sidebar.slider(
                field_meta["label"],
                min_value=float(field_meta["min"]),
                max_value=float(field_meta["max"]),
                value=float(field_meta["value"]),
                step=float(field_meta["step"]),
                key=f"slider_{field_name}",
            )
        return input_features

    current_inputs = get_input_features()
    sdb_col1, sdb_col2 = st.sidebar.columns(2)
    with sdb_col1:
        predict_button = st.sidebar.button("Assess Risk", key="predict")
    with sdb_col2:
        reset_button = st.sidebar.button("Reset", key="clear")
    if predict_button:
        st.session_state['input_features'] = current_inputs
    if reset_button:
        st.session_state['input_features'] = {}
        for field_name, field_meta in FIELD_CONFIG.items():
            st.session_state[f"slider_{field_name}"] = field_meta["value"]
    return None

def app_body():
    st.title("Breast Cancer Screening Risk Checker")
    st.caption("A Streamlit deployment built from a locally trained scikit-learn classifier.")

    hero_col, info_col = st.columns([1.3, 1])
    with hero_col:
        st.markdown(
            """
            Use the sidebar to enter seven cell-level features.
            The model estimates whether the pattern is more consistent with a malignant case that should be escalated for further screening.
            """
        )
    with info_col:
        st.info(
            "This tool is for workshop demonstration only. It is not a medical device and must not be used for real clinical decisions."
        )

    if st.session_state['input_features']:
        assessment = get_prediction(**st.session_state['input_features'])
        probability = assessment["malignant_probability"]

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Malignant Probability", f"{probability:.1%}")
        with metric_col2:
            risk_band = "High" if probability >= 0.6 else "Moderate" if probability >= 0.35 else "Low"
            st.metric("Risk Band", risk_band)

        if assessment["prediction"] == 1:
            st.error("System assessment says: suspicious pattern detected. Recommend further screening.")
        else:
            st.success("System assessment says: lower-risk pattern detected.")

        st.subheader("Submitted Feature Values")
        st.dataframe(
            [st.session_state['input_features']],
            use_container_width=True,
        )
    else:
        st.markdown("Select values in the sidebar and click `Assess Risk` to run the model.")
    return None

def main():
    st.set_page_config(page_title="DSSI Cancer Screening Checker", page_icon=":stethoscope:", layout="wide")
    app_sidebar()
    app_body()
    return None

if __name__ == "__main__":
    main()
