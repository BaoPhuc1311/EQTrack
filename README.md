# 📈 **EQ Track**

## 🧠 **Introduction**  
**EQ Track** is a machine learning project focused on predicting emotional intelligence (EQ) based on various personal and psychological factors. The project uses the Random Forest model to classify individuals based on their EQ levels, applying data preprocessing techniques such as label encoding, scaling, and feature importance analysis.

## 🎯 **Objective**  
- Collect and preprocess datasets related to emotional intelligence and mental health.  
- Train a machine learning model (Random Forest) to predict emotional intelligence (EQ) based on available features.  
- Explore important psychological factors such as depression, stress, sleep patterns, and academic/work pressure.  
- Visualize the performance of the model and feature importance.  
- Provide insights into how various factors impact emotional intelligence and suggest potential interventions.

## 📦 **Requirements**  
- Python 3.x  
- Pandas, NumPy, Matplotlib/Seaborn, Scikit-learn  
- joblib (for saving the model)

## 📊 **Data**  
This project uses the following dataset:  
- [Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)

## 🛠️ **Installation & Usage**  

1. **Clone the repository and set up a virtual environment**:
    ```bash
    git clone https://github.com/BaoPhuc1311/EmotionPredict.git
    cd EmotionPredict
    python -m venv venv
    .\venv\Scripts\activate
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the `main.py` script**:
    ```bash
    python main.py
    ```

## 📄 **License**  
This project is licensed under the [MIT License](LICENSE).
