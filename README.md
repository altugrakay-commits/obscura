# 🛰️ Obscura: Physics-Informed Industrial Emission Predictor

**🔴 Live Interactive Dashboard:** [https://obscura-efud2zwezcafebqzbfxjtv.streamlit.app/](https://obscura-efud2zwezcafebqzbfxjtv.streamlit.app/)

## Overview
This project was developed for the **OBSCURA Hackathon**. It is a PyTorch-based multi-modal deep learning pipeline designed to predict five industrial air pollutants (CH₄, NO₂, CO, AAI, and FIRMS) directly from multi-spectral Sentinel-2 satellite imagery and meteorological data.

Read the full research paper here: [obscura_paper.pdf](./obscura_paper.pdf)

## 🧠 The Physics-Informed Approach
Traditional computer vision models treat satellite images as isolated pixel grids. As a Ph.D. in Materials Science & Metallurgy, I approached this problem by embedding **combustion thermodynamics and atmospheric dispersion physics** directly into the neural network architecture. 

### Key Innovations:
1. **Masked MSE Loss:** Methane (CH₄) data is missing in 65% of the training set due to cloud cover. A custom masked loss function computes gradients only on the targets present, allowing the model to train on all 3,218 samples.
2. **Physics-Informed Penalty:** A custom regularization term penalizes physically impossible predictions. If the model predicts high Fire Intensity but zero Smoke/Dust, the loss function applies a penalty, as this violates basic thermodynamic principles.
3. **RAM Caching Pipeline:** A custom PyTorch Dataset class caches the 14-band `.tif` images in RAM, reducing training epoch times from 200 seconds to 25 seconds.

## 🖥️ Interactive Dashboard
The repository includes `app.py`, a Streamlit dashboard that allows users to upload a Sentinel-2 image, adjust weather sliders, and run inference. It features a **Metallurgical Physics Constraint Check** that visually flags thermodynamically inconsistent predictions in real-time.

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/obscura_hackathon.git
   cd obscura_hackathon
   ```
2. Create a virtual environment and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place the `images/` folder, `train.csv`, and `test.csv` into a `data/` directory
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```