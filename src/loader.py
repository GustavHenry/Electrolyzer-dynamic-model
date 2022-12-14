from abc import ABC, abstractmethod
import pandas as pd
import os
from keys import CACHE_DIR, COMPRESSION, LoaderType, DataDir
import pickle
from datetime import date
import numpy as np

class Loader(ABC):
    def __init__(
        self, fp_cache, use_cahce=True, data_type=LoaderType.dataframe
    ) -> None:
        """这里只支持dataframe和model类"""
        super().__init__()
        self.fp_cache = os.path.join(CACHE_DIR, fp_cache)
        self.use_cache = use_cahce
        self.data_type = data_type

    @abstractmethod
    def run(self):
        """load and process required data from files"""
        raise NotImplementedError

    def cache(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        if self.data_type == LoaderType.dataframe:
            self.data.to_pickle(self.fp_cache, compression=COMPRESSION)
        else:
            with open(self.fp_cache, "wb") as f:
                pickle.dump(self.data, f, protocol=4)

    def clear_cache(self):
        if os.path.exists(self.fp_cache):
            os.remove(self.fp_cache)
            print("    cache removed: " + self.fp_cache)

    def load(self):
        if self.use_cache and os.path.exists(self.fp_cache):
            print("   Reading data from cache: {}".format(self.fp_cache))
            if self.data_type == LoaderType.dataframe:
                self.data = pd.read_pickle(self.fp_cache, compression=COMPRESSION)
            else:
                with open(self.fp_cache, "rb") as f:
                    self.data = pickle.load(f)
        else:
            self.data = self.run()
            if self.use_cache:
                self.cache()

        return self.data
    
    @staticmethod
    def save_model(
        model,
        file_name,
        score,
        date = date.today(),
        folder = DataDir.Model_thermal_model,
    ):
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = os.path.join(
            folder,
            '_'.join([
                file_name,str(np.round(score,2)),date.strftime('%Y%m%d')
            ])
        )
        with open(file_path,'wb') as f:
            pickle.dump(
                model,
                f,
                protocol=4
            )
        print('Model saved to ' + file_path)
    
    @staticmethod
    def load_model(
        file_name,
        folder = DataDir.Model_thermal_model,
    ):
        file_path = os.path.join(
            folder,file_name
        )
        if os.path.exists(file_path):
            with open(file_path,'rb') as f:
                model = pickle.load(f)
            return model
        else:
            raise FileNotFoundError

