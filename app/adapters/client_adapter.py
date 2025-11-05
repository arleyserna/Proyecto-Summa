import pandas as pd
import numpy as np
import joblib
from app.utils.config import ENCODERS_PATH


class ClientDataAdapter:
    
    def __init__(self, client_data):
        # Convert single dictionary to DataFrame with one row
        self.client_data = pd.DataFrame([client_data])
        self.le_target = joblib.load(ENCODERS_PATH)
        self.numeric_features = ['Charges', 'Demand']



    def get_summary(self) -> pd.DataFrame:
        """
        Provide a summary of the client data.

        :return: A DataFrame containing the summary of the client data.
        """
        return self.client_data.head()

        
    def get_features(self) -> pd.DataFrame:
        """
        Extract and return the features from the client data.

        :return: A DataFrame containing the features.
        """
        # Return all features except Class and Demand which are target variables
        #features = self.client_data.drop(['Class', 'Demand'], axis=1, errors='ignore')
        #return features
        pass

    
    def convert_to_model_format(self) -> pd.DataFrame:
        """
        Convert the client data into a format suitable for prediction.
        :return: A DataFrame ready for prediction.
        """
        data = self.client_data.copy()

        # Drop ID field and target variables if they exist
        features_to_drop = ['autoID', 'Class']
        data = data.drop(features_to_drop, axis=1, errors='ignore')
        

        data['Demand'] = pd.to_numeric(data['Demand'], errors='coerce').fillna(0)
        data['Charges'] = pd.to_numeric(data['Charges'], errors='coerce').fillna(0)

        self.categorical_features = [col for col in data.columns if col not in self.numeric_features]

        for val in self.le_target.values():
            print(f"Loaded Label Encoder classes: {val.classes_} ")
        

        # Convert categorical variables to numeric
        for col in self.categorical_features:
                le = self.le_target[col]
                data[col] = le.transform(data[col].astype(str))

                print(f"Label Encoder for {col}: {le.classes_} ")

        data = pd.get_dummies(data, columns=self.categorical_features, drop_first=False)
        data = data.reindex(columns=['Charges','Demand','SeniorCity_1','Partner_1','Dependents_1','Service1_1','Service2_1','Service2_2','Security_1',
                                     'Security_2','OnlineBackup_1','OnlineBackup_2','DeviceProtection_1','DeviceProtection_2','TechSupport_1','TechSupport_2',
                                     'Contract_1','Contract_2','PaperlessBilling_1','PaymentMethod_1','PaymentMethod_2','PaymentMethod_3'], fill_value=0)
        data.info()

        return data

