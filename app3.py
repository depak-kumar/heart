import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Optionally, you can add more eded
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the heart disease dataset (you can replace this with your dataset)
data = pd.read_csv("/content/drive/MyDrive/heart.csv")


# Create a Streamlit web app

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier (you can replace this with your model)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Create a Streamlit web app
st.title("Heart Disease Prediction App")
st.image("https://th.bing.com/th/id/R.89f518358227a5ab591a63e971b4b9b5?rik=%2fff3CI6%2bvmpElw&riu=http%3a%2f%2fwww.interactive-biology.com%2fwp-content%2fuploads%2f2012%2f05%2fIllustration-of-the-Human-heart.jpg&ehk=%2f2une2rnSa4SqFSShHlRkGAPR%2br1m1FFeZo9VhhReiA%3d&risl=&pid=ImgRaw&r=0", width=300, caption="Heart Disese ")
st.subheader("CSV File Viewer")
if st.button("Show DATA FILE"):
# Display the CSV data in a table format
  st.dataframe(data)
# Add a sidebar with user input parameters
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 18,2)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 126, 564, 240)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 71, 202, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 1)
    thal = st.sidebar.selectbox("Thalassemia Type", ["Normal", "Fixed Defect", "Reversible Defect"])

    


    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    cp_encoded = 0  # Initialize as 0
    if cp == "Atypical Angina":
        cp_encoded = 1
    elif cp == "Non-Anginal Pain":
        cp_encoded = 2
    elif cp == "Asymptomatic":
        cp_encoded = 3

    restecg_encoded = 0  # Initialize as 0
    if restecg == "ST-T Wave Abnormality":
        restecg_encoded = 1
    elif restecg == "Left Ventricular Hypertrophy":
        restecg_encoded = 2

    slope_encoded = 0  # Initialize as 0
    if slope == "Flat":
        slope_encoded = 1
    elif slope == "Downsloping":
        slope_encoded = 2

    thal_encoded = 0  # Initialize as 0
    if thal == "Fixed Defect":
        thal_encoded = 1
    elif thal == "Reversible Defect":
        thal_encoded = 2

    return [age, sex, cp_encoded, trestbps, chol, fbs, restecg_encoded, thalach, exang, oldpeak, slope_encoded, ca, thal_encoded]

user_input = user_input_features()

# Display the user inputs
st.subheader("User Input:")
# st.subheader("Age :")
st.write("Age:", user_input[0])
st.write("Sex:", "Male" if user_input[1] == 1 else "Female")
st.write("Chest Pain Type:", user_input[2])
st.write("Resting Blood Pressure (mm Hg):", user_input[3])
st.write("Cholesterol (mg/dl):", user_input[4])
st.write("Fasting Blood Sugar > 120 mg/dl:", "Yes" if user_input[5] == 1 else "No")
st.write("Resting Electrocardiographic Results:", user_input[6])
st.write("Maximum Heart Rate Achieved:", user_input[7])
st.write("Exercise Induced Angina:", "Yes" if user_input[8] == 1 else "No")
st.write("ST Depression Induced by Exercise Relative to Rest:", user_input[9])
st.write("Slope of the Peak Exercise ST Segment:", user_input[10])
st.write("Number of Major Vessels Colored by Fluoroscopy:", user_input[11])
st.write("Thalassemia Type:", user_input[12])
# Sample dataset (you can replace this with your own dataset)
data = pd.read_csv("/content/drive/MyDrive/heart.csv")

# Create a Streamlit web app
st.title("Feature Visualization")
st.header("Visualization of Selected Feature")
# Display the first few rows of the dataset
st.subheader("Sample Data")
st.write(data.head())

#Sidebar to select features
selected_feature = st.sidebar.selectbox("Select a Feature to Visualize", data.columns)

    # Sidebar to choose plot type
plot_type = st.sidebar.radio("Select Plot Type", ["Histogram", "Bar Plot", "Box Plot", "Scatter Plot"])

# Visualize the selected feature
if plot_type == "Histogram":
    st.subheader("Histogram")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=data, x=selected_feature, bins=20, kde=True, ax=ax)
    st.pyplot(fig)

elif plot_type == "Bar Plot":
    st.subheader("Bar Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=data, x=selected_feature, ax=ax)
    st.pyplot(fig)

elif plot_type == "Box Plot":
    st.subheader("Box Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=data, x=selected_feature, y="age", ax=ax)
    st.pyplot(fig)

elif plot_type == "Scatter Plot":
    st.subheader("Scatter Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x="age", y=selected_feature, hue="trestbps", ax=ax)
    st.pyplot(fig)

# Predict the target using the trained model
prediction = clf.predict([user_input])

st.subheader("Prediction:")
if st.button("Predict ") :
  if prediction[0] == 0:
    st.success("Prediction Sucessfull")
    st.write("No Heart Disease")
  elif prediction[0] == 1:
    st.success("Prediction Sucessfull")
    st.write("Heart Disease")

# Display model performance
st.subheader("Model Performance:")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))


st.subheader("Made By DEEPAK KUAMR")


# Add more visualizations as needed

