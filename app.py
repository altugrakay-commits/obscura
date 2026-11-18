import streamlit as st
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import time
import io
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler

# --- 1. MODEL ARCHITECTURE ---
class ObscuraMultiTaskModel(nn.Module):
    def __init__(self, tabular_input_dim, num_targets=5):
        super(ObscuraMultiTaskModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(14, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        self.mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + 32, 64), nn.ReLU(), nn.Linear(64, num_targets)
        )

    def forward(self, img, tabular):
        img_features = self.cnn(img).view(img.size(0), -1)
        tab_features = self.mlp(tabular)
        combined = torch.cat((img_features, tab_features), dim=1)
        return self.fc(combined)

# --- 2. LOAD MODEL & CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = 'data'
TARGETS = ['t_ch4', 't_no2', 't_co', 't_aai', 't_firms']

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
features = [c for c in train_df.columns if c not in TARGETS and c not in ['sample_id', 'site_key', 'image_id']]

scaler = StandardScaler()
scaler.fit(train_df[features].fillna(0))
feature_means = train_df[features].mean().fillna(0)

model = ObscuraMultiTaskModel(tabular_input_dim=len(features)).to(DEVICE)
try:
    model.load_state_dict(torch.load('obscura_model.pth', map_location=DEVICE))
    model.eval()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load model. Error: {e}")
    model_loaded = False

# --- 3. APP UI ---
st.set_page_config(page_title="Obscura: Emissions AI", layout="wide", page_icon="🛰️")
st.title("🛰️ Obscura: Industrial Emission Predictor")
st.markdown("### Physics-Informed Multi-Task Learning for Satellite Emissions")

st.sidebar.header("Simulate Facility Data")
uploaded_image = st.sidebar.file_uploader("Upload a Sentinel-2 Image (.tif)", type=["tif"])
temperature = st.sidebar.slider("Ambient Temperature (°C)", -20.0, 50.0, 25.0)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0)

if uploaded_image is not None and model_loaded:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Satellite Image Preview")
        uploaded_image.seek(0)
        img_array = tiff.imread(io.BytesIO(uploaded_image.read()))
        
        if not isinstance(img_array, np.ndarray): img_array = np.array(img_array[0])
        if img_array.shape[0] < img_array.shape[-1]: img_array = np.transpose(img_array, (1, 2, 0))
            
        rgb = img_array[:, :, [2, 1, 0]]
        rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) 
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb_norm)
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        st.subheader("Model Inference")
        if st.button("Run Physics-Informed Prediction", type="primary"):
            with st.spinner("Running CNN + MLP forward pass..."):
                
                # 1. Process Image
                img_tensor = torch.tensor(img_array, dtype=torch.float32)
                if img_tensor.shape[0] != 14: img_tensor = img_tensor.permute(2, 0, 1) 
                img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE) 
                
                # 2. Process Tabular Data
                tab_vector = feature_means.copy()
                if 'temperature' in tab_vector: tab_vector['temperature'] = temperature
                if 'wind_speed' in tab_vector: tab_vector['wind_speed'] = wind_speed
                
                tab_scaled = scaler.transform(tab_vector.values.reshape(1, -1))
                tab_tensor = torch.tensor(tab_scaled, dtype=torch.float32).to(DEVICE)
                
                # 3. Predict
                with torch.no_grad():
                    preds_tensor = model(img_tensor, tab_tensor).cpu().numpy()[0]
                
                preds = {
                    "Methane (CH4)": preds_tensor[0],
                    "Nitrogen Dioxide (NO2)": preds_tensor[1],
                    "Carbon Monoxide (CO)": preds_tensor[2],
                    "Smoke/Dust (AAI)": preds_tensor[3],
                    "Fire Intensity (FIRMS)": preds_tensor[4]
                }
                
                # Display metrics
                st.metric("Methane (CH4)", f"{preds['Methane (CH4)']:.3f}")
                st.metric("Nitrogen Dioxide (NO2)", f"{preds['Nitrogen Dioxide (NO2)']:.3f}")
                st.metric("Carbon Monoxide (CO)", f"{preds['Carbon Monoxide (CO)']:.3f}")
                st.metric("Smoke/Dust (AAI)", f"{preds['Smoke/Dust (AAI)']:.3f}")
                st.metric("Fire Intensity (FIRMS)", f"{preds['Fire Intensity (FIRMS)']:.3f}")
                
                # --- PHYSICS INFORMED VALIDATION ---
                st.markdown("---")
                st.subheader("Metallurgical Physics Constraint Check")
                
                passed = True
                
                # Rule 1: Combustion Thermodynamics
                if preds["Fire Intensity (FIRMS)"] > 1.2 and preds["Smoke/Dust (AAI)"] < 0.2:
                    st.error("⚠️ **Combustion Violation Detected!**")
                    st.write(f"High fire predicted ({preds['Fire Intensity (FIRMS)']:.2f}) but low smoke detected ({preds['Smoke/Dust (AAI)']:.2f}). In industrial combustion, high heat without particulate matter is thermodynamically inconsistent.")
                    passed = False
                
                # Rule 2: Atmospheric Dispersion
                elif wind_speed > 12.0 and preds["Smoke/Dust (AAI)"] > 1.2:
                    st.error("⚠️ **Dispersion Violation Detected!**")
                    st.write(f"High wind speed ({wind_speed} m/s) should rapidly disperse airborne particulates, yet the model predicts heavy smoke ({preds['Smoke/Dust (AAI)']:.2f}). This violates Gaussian plume dispersion models.")
                    passed = False
                
                if passed:
                    st.success("✅ **Physics Constraints Passed**")
                    st.write("The emission ratios align with expected combustion thermodynamics and atmospheric dispersion models.")