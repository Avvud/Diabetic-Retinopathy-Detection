# AI-Assisted Ophthalmology: Diabetic Retinopathy Detection

This project is an **AI-Assisted Ophthalmology Platform** designed for the automated detection and clinical analysis of **Diabetic Retinopathy (DR)**. By leveraging an ensemble of state-of-the-art deep learning architectures—specifically **EfficientNet-B3**, **DenseNet121**, and **ResNet50**—the system provides high-accuracy classification across five clinical severity stages, from "No DR" to "Proliferative DR."

Beyond raw classification, the platform prioritizes **transparency and explainability** through integrated **Grad-CAM visualisations**. These heatmaps highlight the specific retinal regions, such as microaneurysms or hemorrhages, that influenced the model's decision, allowing clinicians to verify AI findings against visual evidence. 

The application further bridges the gap between AI inference and clinical practice by integrating **LLaMA 3.2** (via Ollama). This component synthesizes the image analysis with user-provided patient history to generate structured clinical summaries and risk assessments. This "Human-AI" collaborative approach ensures that diagnostic results are contextualized within the patient's broader clinical profile. 

Built with a premium **Gradio web interface**, the system offers a streamlined workflow for clinicians to upload fundus images, review visual activations, and receive immediate diagnostic support, making it a robust tool for ophthalmological research and decision-making.

## Features

- **Ensemble Inference**: Combines multiple CNN architectures for robust predictions.
- **Explainable AI**: Real-time Grad-CAM heatmaps for diagnostic transparency.
- **LLM Clinical Summary**: Automated generation of clinical reports using LLaMA 3.2.
- **Modern UI**: Dark-themed, responsive interface designed for clinical environments.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Ollama**:
   - Install Ollama and pull the LLaMA 3.2 model:
     ```bash
     ollama pull llama3.2
     ```
3. **Run the App**:
   ```bash
   python app.py
   ```

## Disclaimer

This tool is for **research and clinical decision support only**. It does not replace professional ophthalmological diagnosis. Final clinical decisions must be made by a qualified healthcare professional.
