import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Hàm kiểm tra giá trị cột
def check_column_values(df, column):
    print(f"Unique values in {column}:", df[column].unique())

try:
    # Đọc dữ liệu
    data = pd.read_csv("data/student_depression_dataset.csv")
    print("Data loaded successfully!")
    print(data.head())
    print("\nColumns:", data.columns.tolist())
    print("\nInfo:")
    print(data.info())
    print("\nMissing values:")
    print(data.isnull().sum())

    # Kiểm tra giá trị của các cột object
    object_cols = data.select_dtypes(include=['object']).columns
    for col in object_cols:
        check_column_values(data, col)

    # Làm sạch cột Sleep Duration
    def map_sleep_duration(value):
        if '5-6 hours' in value:
            return 5.5
        elif 'Less than 5 hours' in value:
            return 4.0
        elif '7-8 hours' in value:
            return 7.5
        elif 'More than 8 hours' in value:
            return 9.0
        else:  # 'Others'
            return 6.0  # Giá trị trung bình giả định
    data['Sleep Duration'] = data['Sleep Duration'].apply(map_sleep_duration)

    # Làm sạch cột Financial Stress
    data['Financial Stress'] = data['Financial Stress'].replace('?', data['Financial Stress'].mode()[0])
    data['Financial Stress'] = pd.to_numeric(data['Financial Stress'], errors='coerce').fillna(data['Financial Stress'].mode()[0])

    # Sửa tên cột dài để dễ xử lý
    data = data.rename(columns={'Have you ever had suicidal thoughts ?': 'Suicidal Thoughts'})

    # Mã hóa cột phân loại
    categorical_cols = ['Gender', 'City', 'Profession', 'Dietary Habits', 'Degree', 
                        'Family History of Mental Illness', 'Suicidal Thoughts']
    le = LabelEncoder()
    for col in categorical_cols:
        try:
            data[col] = le.fit_transform(data[col])
        except Exception as e:
            print(f"Error encoding {col}: {e}")

    # Tạo nhãn EQ
    data['EQ_label'] = data['Depression'].apply(lambda x: 1 if x == 0 else 0)  # 1: EQ cao, 0: EQ thấp

    # Kiểm tra phân phối nhãn
    print("\nEQ_label distribution:")
    print(data['EQ_label'].value_counts(normalize=True))

    # Chọn đặc trưng (X) và nhãn (y)
    X = data.drop(['id', 'Depression', 'EQ_label'], axis=1)
    y = data['EQ_label']

    # Chuẩn hóa các cột số
    numeric_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 
                    'Job Satisfaction', 'Sleep Duration', 'Work/Study Hours', 'Financial Stress']
    scaler = StandardScaler()
    try:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    except Exception as e:
        print(f"Error in scaling: {e}")

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá
    y_pred = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Tầm quan trọng đặc trưng
    print("\nFeature Importance:")
    importances = model.feature_importances_
    feature_names = X.columns
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")

    # Lưu mô hình
    import joblib
    joblib.dump(model, 'rf_model.pkl')
    print("\nModel saved as rf_model.pkl")

except FileNotFoundError:
    print("Error: File 'data/student_depression_dataset.csv' not found. Please check the file path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")