import re

_path_to_env=None
_cache = None

class Config:
    _path_to_env = None
    _cache = None
    _cache_dict = {}


    @classmethod
    def reset(cls):
        cls._path_to_env = None
        cls._cache = None
        cls._cache_dict = {}

    @classmethod
    def get(cls,key,default=None):
        if Config._cache is None:
            with open(cls._path_to_env,"r") as f:
                cls._cache = f.read()
        if key not in cls._cache_dict:
            ll = re.findall(f"\s*{key}\s*=\s*(\S+)",cls._cache)
            if ll:
                cls._cache_dict[key] = ll[-1]
            else:
                return default
        return cls._cache_dict[key]

def set_path(path):
    Config.reset()
    Config._path_to_env = path

def get(key,default=None):
    return Config.get(key,default)

# MLFLOW_URI = "https://mlflow.lewagon.co/"
# bucket name - replace with your GCP bucket name
# BUCKET_NAME="wagon-data-789-conti"
# BUCKET_TRAIN_DATA_PATH ='data/train_1k.csv'
# STORAGE_LOCATION = 'models/taxifare/'







