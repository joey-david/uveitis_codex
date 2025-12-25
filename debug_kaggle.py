import kaggle
import os

os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

api = kaggle.KaggleApi()
api.authenticate()

files = api.dataset_list_files('andreivann/eyepacs').files
if files:
    f = files[0]
    print("Attributes:", dir(f))
    print("String rep:", str(f))
    try:
        print("Size:", f.size)
    except Exception as e:
        print("Error accessing size:", e)
    
    try:
        print("totalBytes:", f.totalBytes)
    except Exception as e:
        print("Error accessing totalBytes:", e)
