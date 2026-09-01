import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Data/Cardiovascular_Disease.csv')

df['age'] = df['age']//365

hw = df[
    (df['height'].between(136,200)) & 
    (df['weight'].between(35,120)) & 
    (df['ap_hi'].between(90,200)) & 
    (df['ap_lo'].between(50, 100))
]
df = hw

def logistic_cardio_predict(): ## make svm cardio predict hw
    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    target = 'cardio'

    X = df[features]
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train) # Learning + implement
    X_test_scale = scaler.transform(X_test)

    model = LogisticRegression(
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train_scale, Y_train)

    Y_pred = model.predict(X_test_scale)

    cr = classification_report(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)

    return features, scaler, model, Y_pred, cr, cm

from sklearn.svm import SVC
def svm_cardio_predict():
    df_sample = df.sample(n=5000, random_state=42)

    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    target = 'cardio'

    X = df_sample[features]
    Y = df_sample[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train) # Learning + implement
    X_test_scale = scaler.transform(X_test) # implement

    model = SVC()
    model.fit(X_train_scale, Y_train)
    Y_pred = model.predict(X_test_scale)

    cr = classification_report(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)

    return features, scaler, model, Y_pred, cr, cm