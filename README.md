# 📊 CYBER-ML: Neural Network Prediction Engine
<div align="center">
  <h3>A cyberpunk-themed machine learning application for predictive analytics</h3>
</div>
**AF3005 – Programming for Finance**  

📍 **FAST National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad**  
👨‍🏫 **Instructor:** Dr. Usama Arshad (Assistant Professor, FSM)  
🎓 **Program:** BS Financial Technology (BSFT)  
📅 **Semester:** Spring 2025  
📌 **Sections:** BSFT06A, BSFT06B, BSFT06C  

---

## 📌 Overview

CYBER-ML is a streamlined, cyberpunk-themed machine learning application built with Streamlit. It provides an intuitive interface for data upload, preprocessing, model training, and evaluation - all within a visually stunning neon interface. Perfect for both ML beginners and experienced data scientists who want to quickly build and validate regression models without writing code.

## ✨ Features

- **🔮 Full ML Pipeline**: End-to-end machine learning from data upload to prediction
- **🧪 Advanced Preprocessing**: Multiple options for data cleaning and transformation
- **🤖 Multiple ML Algorithms**: Choose between Linear Regression and Random Forest models
- **📈 Interactive Visualizations**: Beautiful cyberpunk-styled data visualizations
- **📊 Model Evaluation**: Comprehensive model performance metrics
- **💾 Export Results**: Download predictions and model results

## 🛠️ Installation

```bash
# Clone this repository
git clone https://github.com/your-username/cyber-ml.git
cd cyber-ml

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements

The application requires the following Python packages:

- streamlit==1.31.0
- pandas==2.1.2
- numpy==1.26.2
- plotly==5.18.0
- scikit-learn==1.3.2
- matplotlib==3.8.2
- openpyxl==3.1.2
- pillow==10.1.0
- scipy==1.11.3

## 🚀 Usage

1. Launch the application:

```bash
streamlit run app.py
```

2. Access the app in your browser (usually at http://localhost:8501)

3. Follow the guided workflow:
   - Upload your dataset (CSV or Excel)
   - Select features and target variable
   - Apply preprocessing techniques
   - Train your machine learning model
   - Evaluate performance and visualize results

## 🔋 ML Pipeline Steps

### 1️⃣ Data Upload & Selection
Upload your CSV or Excel file and select the target variable and features for prediction.

### 2️⃣ Data Preprocessing
Multiple preprocessing options available:
- **Missing Value Handling**: Mean, median, most frequent, or constant imputation
- **Outlier Detection**: Clip or remove outliers using IQR method
- **Feature Transformation**: Log or square root transformations
- **Feature Scaling**: Standardization, Min-Max scaling, or Robust scaling

### 3️⃣ Data Analysis
Interactive visualizations of feature-target relationships to help understand your data.

### 4️⃣ Model Training
Train your selected machine learning model (Linear Regression or Random Forest) with customizable parameters.

### 5️⃣ Model Evaluation
Comprehensive evaluation with metrics like RMSE, R² score, and visualizations comparing actual vs. predicted values.

## 🎮 Customization

The application provides several customization options:

- **Algorithm Selection**: Choose between Linear Regression and Random Forest
- **Test Size Ratio**: Control the split between training and testing data
- **Preprocessing Options**: Multiple techniques for handling data issues
- **Visualization Options**: Select features to visualize and analyze

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Streamlit for the amazing web framework
- Plotly for the interactive visualizations
- Scikit-learn for the machine learning tools
- All contributors and users of this project

---

<div align="center">
  <p>[ CYBER-ML v2.0.77 ]</p>
  <p>© 2025 • CYBER FINANCE ML</p>
</div>
