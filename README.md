# EM-CNN-pipeline-for-3D-liver-tumor-detection.
Hybrid EM-CNN framework for 3D liver tumor detection combining electromagnetic wave simulation with deep learning-based segmentation.

# Hybrid EM-CNN Liver Tumor Detection

This project presents a hybrid medical imaging framework that combines electromagnetic (EM) wave simulation with deep learning-based 3D segmentation for improved liver tumor detection.

##  Overview

Traditional CNN-based segmentation models often struggle with detecting small or low-contrast tumors due to class imbalance and limited visibility in CT scans. 

To address this, we introduce a physics-guided pipeline where:

- Electromagnetic wave simulations are used to detect dielectric anomalies in tissue
- These anomalies are used to generate regions of interest (ROI)
- A 3D UNet model performs high-precision segmentation on these regions

##  Key Idea

EM waves are sensitive to dielectric contrast between healthy liver tissue and tumors. Even when tumors are small or hard to detect in CT scans, they can cause measurable disturbances in EM field propagation.

We leverage this to guide the CNN model towards suspicious regions, improving tumor detection reliability.

##  Pipeline

EM Simulation (ANSYS HFSS)
→ Field Distortion Analysis
→ ROI Extraction
→ CT Patch Extraction
→ 3D UNet Segmentation
→ Fusion & Validation

##  Model Performance

- Liver Dice: ~0.68  
- Tumor Dice: ~0.62  
- Accuracy: ~95%  

The model demonstrates strong segmentation performance, with the EM-guided pipeline designed to further improve detection of small tumors.

##  Tech Stack

- PyTorch (3D UNet)
- ANSYS HFSS (EM simulation)
- NumPy / SciPy
- Medical imaging datasets (LiTS-based)

##  Research Motivation

This work explores a hybrid approach combining physics-based sensing with data-driven learning, aiming to improve robustness in tumor detection where conventional CNNs fall short.

##  Future Work

- EM-to-CT coordinate alignment optimization  
- Multimodal fusion learning (EM + CT as input)  
- Improved tumor recall using adaptive ROI sampling  
- Real-world validation with noisy EM data  

## 📎 Note

This project is a research-oriented prototype demonstrating the feasibility of integrating EM simulation with deep learning for medical imaging.

## How to Run

1. Download dataset:
   https://www.kaggle.com/datasets/nazarhussain114/liver-tumor-classification-and-segmentation

2. Place it as:
   archive/Task03_Liver/

3. Train:
   python train_3d.py --data_root archive

4. Evaluate:
   python bestmodel.py

   
##  Results

- Liver Dice: ~0.68  
- Tumor Dice: ~0.62  
- Accuracy: ~95%


  Model weights and dataset are not included due to size constraints.

##  Sample Output
<img width="1092" height="897" alt="Screenshot 2026-04-07 151031" src="https://github.com/user-attachments/assets/afde0ae9-1bf5-4429-b35b-83c2e1b0662b" />


## best epoch -
<img width="1082" height="59" alt="Screenshot 2026-04-07 151119" src="https://github.com/user-attachments/assets/4553fe1c-8b6c-45c5-a43c-6728bed6b129" />




