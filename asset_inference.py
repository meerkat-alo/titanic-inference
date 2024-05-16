#asset_[step_name].py
 
# -*- coding: utf-8 -*-
import os
import sys
from alolib.asset import Asset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd 
from titanic_source import TITANIC
#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------
class UserAsset(Asset):
    def __init__(self, asset_structure):
        super().__init__(asset_structure)
        self.args       = self.asset.load_args()
        self.config     = self.asset.load_config()
        self.data       = self.asset.load_data() 
 
    @Asset.decorator_run
    def run(self):
        df = self.data['dataframe0']
        x_columns = self.config['x_columns'] # from input asset saved config
        
        titanic = TITANIC(self.asset)
        predicted_class, predict_proba = titanic.inference(df, x_columns)
        # concat input data and output (predicted) for re-training later 
        self.data['output'] = pd.concat([df, pd.DataFrame(predicted_class, columns=['predicted'])], axis=1)
        self.data['probability'] = predict_proba
        
        self.asset.save_data(self.data)
        self.asset.save_config(self.config)
 
 
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    envs, argv, data, config = {}, {}, {}, {}
    ua = UserAsset(envs, argv, data, config)
    ua.run()
