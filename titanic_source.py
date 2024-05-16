import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle 

class TITANIC:
    def __init__(self, asset):
        self.asset = asset # for using alolib API
    
    def train(self, df, x_columns, y_column): 
        X = pd.get_dummies(df[x_columns])
        y = df[y_column] 
        n_estimators = self.asset.load_args()['n_estimators']

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, random_state=1)
        model.fit(X, y)
        
        # save trained model
        model_path = self.asset.get_model_path()  
        try: 
            with open(model_path + 'random_forest_model.pkl', 'wb') as file:
                pickle.dump(model, file)
        except Exception as e: 
            self.asset.save_error("Failed to save trained model" + str(e)) # error logging
        
        return model_path 

    def inference(self, df, x_columns):
        X = pd.get_dummies(df[x_columns])
        
        # load trained model
        model_path = self.asset.get_model_path()  
        try: 
            with open(model_path + 'random_forest_model.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
        except Exception as e:
            self.asset.save_error("Failed to load trained model" + str(e))
        
        predicted_class = loaded_model.predict(X)
        predict_proba = loaded_model.predict_proba(X)
        
        return predicted_class, predict_proba