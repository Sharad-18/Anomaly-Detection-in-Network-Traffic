import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

# Load your KDD Cup 1999 dataset
@st.cache_data
def load_data():
    df = pd.read_csv(
        'D:/Chrome-downloads/kddcup.data/kddcup.data',
        header=None,
        names=[
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'Label'
        ]
    )
    df.drop(['num_outbound_cmds'], axis=1, inplace=True)
    return df

from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix

def preprocess_data(df):
    # Limit the sample size for training to reduce memory usage
    sample_size = 100000  # Adjust based on your memory constraints
    df_sample = df.sample(n=sample_size, random_state=42)
    
    df_normal = df_sample[df_sample['Label'] == 'normal.']
    df_abnormal = df_sample[df_sample['Label'] != 'normal.']
    
    df_abnormal = df_abnormal.sample(min(1000000 - df_normal.shape[0], df_abnormal.shape[0]), random_state=42)
    df = pd.concat([df_normal, df_abnormal]).sample(frac=1, random_state=42)

    x = df.drop('Label', axis=1)
    y = np.where(df['Label'] == 'normal.', 1, 0)
    
    categorical_columns = ["protocol_type", "service", "flag"]
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", ordinal_encoder, categorical_columns),
        ],
        remainder="passthrough",
    )
    x = preprocessor.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test, preprocessor


# Train the Isolation Forest model
def train_iforest(x_train):
    iforest = IsolationForest(n_estimators=100, bootstrap=True, random_state=42)
    iforest.fit(x_train)
    return iforest

# Streamlit app code
st.title("KDD Cup 1999 Anomaly Detection")

# Display dataset information
st.subheader("Dataset Overview")
df = load_data()
st.write(df.head())

x_train, x_test, y_train, y_test, preprocessor = preprocess_data(df)
iforest = train_iforest(x_train)

y_pred = iforest.predict(x_test)
y_pred = np.where(y_pred == 1, 1, 0)  # normal = 1, anomaly = 0
y_score = iforest.decision_function(x_test)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Plot ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
roc_display.plot(ax=ax)
st.pyplot(fig)

# Add a form for user input
st.subheader("Predict New Input")

with st.form("predict_form"):
    st.write("Enter the features for prediction:")
    
    duration = st.number_input("Duration", value=0)
    protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
    service = st.selectbox("Service", ["http", "smtp", "pop3", "other"])
    flag = st.selectbox("Flag", ["SF", "S0", "REJ", "other"])
    src_bytes = st.number_input("Source Bytes", value=0)
    dst_bytes = st.number_input("Destination Bytes", value=0)
    land = st.number_input("Land", value=0)
    wrong_fragment = st.number_input("Wrong Fragment", value=0)
    urgent = st.number_input("Urgent", value=0)
    hot = st.number_input("Hot", value=0)
    num_failed_logins = st.number_input("Number of Failed Logins", value=0)
    logged_in = st.number_input("Logged In", value=0)
    num_compromised = st.number_input("Number Compromised", value=0)
    root_shell = st.number_input("Root Shell", value=0)
    su_attempted = st.number_input("SU Attempted", value=0)
    num_root = st.number_input("Number of Root", value=0)
    num_file_creations = st.number_input("Number of File Creations", value=0)
    num_shells = st.number_input("Number of Shells", value=0)
    num_access_files = st.number_input("Number of Access Files", value=0)
    num_outbound_cmds = st.number_input("Number of Outbound Commands", value=0)
    is_host_login = st.number_input("Is Host Login", value=0)
    is_guest_login = st.number_input("Is Guest Login", value=0)
    count = st.number_input("Count", value=0)
    srv_count = st.number_input("Srv Count", value=0)
    serror_rate = st.number_input("Serror Rate", value=0.0)
    srv_serror_rate = st.number_input("Srv Serror Rate", value=0.0)
    rerror_rate = st.number_input("Rerror Rate", value=0.0)
    srv_rerror_rate = st.number_input("Srv Rerror Rate", value=0.0)
    same_srv_rate = st.number_input("Same Srv Rate", value=0.0)
    diff_srv_rate = st.number_input("Diff Srv Rate", value=0.0)
    srv_diff_host_rate = st.number_input("Srv Diff Host Rate", value=0.0)
    dst_host_count = st.number_input("Dst Host Count", value=0)
    dst_host_srv_count = st.number_input("Dst Host Srv Count", value=0)
    dst_host_same_srv_rate = st.number_input("Dst Host Same Srv Rate", value=0.0)
    dst_host_diff_srv_rate = st.number_input("Dst Host Diff Srv Rate", value=0.0)
    dst_host_same_src_port_rate = st.number_input("Dst Host Same Src Port Rate", value=0.0)
    dst_host_srv_diff_host_rate = st.number_input("Dst Host Srv Diff Host Rate", value=0.0)
    dst_host_serror_rate = st.number_input("Dst Host Serror Rate", value=0.0)
    dst_host_srv_serror_rate = st.number_input("Dst Host Srv Serror Rate", value=0.0)
    dst_host_rerror_rate = st.number_input("Dst Host Rerror Rate", value=0.0)
    dst_host_srv_rerror_rate = st.number_input("Dst Host Srv Rerror Rate", value=0.0)
    
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame({
            'duration': [duration],
            'protocol_type': [protocol_type],
            'service': [service],
            'flag': [flag],
            'src_bytes': [src_bytes],
            'dst_bytes': [dst_bytes],
            'land': [land],
            'wrong_fragment': [wrong_fragment],
            'urgent': [urgent],
            'hot': [hot],
            'num_failed_logins': [num_failed_logins],
            'logged_in': [logged_in],
            'num_compromised': [num_compromised],
            'root_shell': [root_shell],
            'su_attempted': [su_attempted],
            'num_root': [num_root],
            'num_file_creations': [num_file_creations],
            'num_shells': [num_shells],
            'num_access_files': [num_access_files],
            'num_outbound_cmds': [num_outbound_cmds],
            'is_host_login': [is_host_login],
            'is_guest_login': [is_guest_login],
            'count': [count],
            'srv_count': [srv_count],
            'serror_rate': [serror_rate],
            'srv_serror_rate': [srv_serror_rate],
            'rerror_rate': [rerror_rate],
            'srv_rerror_rate': [srv_rerror_rate],
            'same_srv_rate': [same_srv_rate],
            'diff_srv_rate': [diff_srv_rate],
            'srv_diff_host_rate': [srv_diff_host_rate],
            'dst_host_count': [dst_host_count],
            'dst_host_srv_count': [dst_host_srv_count],
            'dst_host_same_srv_rate': [dst_host_same_srv_rate],
            'dst_host_diff_srv_rate': [dst_host_diff_srv_rate],
            'dst_host_same_src_port_rate': [dst_host_same_src_port_rate],
            'dst_host_srv_diff_host_rate': [dst_host_srv_diff_host_rate],
            'dst_host_serror_rate': [dst_host_serror_rate],
            'dst_host_srv_serror_rate': [dst_host_srv_serror_rate],
            'dst_host_rerror_rate': [dst_host_rerror_rate],
            'dst_host_srv_rerror_rate': [dst_host_srv_rerror_rate]
        })
        
        # Preprocess and predict
        input_data_transformed = preprocessor.transform(input_data)
        prediction = iforest.predict(input_data_transformed)
        prediction_result = "Normal" if prediction[0] == 1 else "Anomaly"
        
        st.write(f"Prediction Result: {prediction_result}")
