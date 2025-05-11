import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the data
df = pd.read_csv('data/f2.csv')

# Encode categorical variables
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df['Soil_Type'] = le_soil.fit_transform(df['Soil_Type'])
df['Crop_Type'] = le_crop.fit_transform(df['Crop_Type'])
df['Fertilizer'] = le_fert.fit_transform(df['Fertilizer'])

# Save encoders for later
with open('model/soil_encoder.pkl', 'wb') as f:
    pickle.dump(le_soil, f)
with open('model/crop_encoder.pkl', 'wb') as f:
    pickle.dump(le_crop, f)
with open('model/fert_encoder.pkl', 'wb') as f:
    pickle.dump(le_fert, f)

# Features and labels
X = df.drop('Fertilizer', axis=1)
y = df['Fertilizer']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('model/fertilizer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")
