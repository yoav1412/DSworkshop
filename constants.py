import os

# the folder containing the Kaggle Data ("Tappy Data", "Archived Users") and MIT Data ("Test Data")
# and to which the output files will be saved also:
DATA_FOLDER = ""

try:
    import localConstants  # if local constants file is available, use a custom DATA_FOLDER path
    DATA_FOLDER = localConstants.DATA_FOLDER
except ImportError:
    localConstants = None
    DATA_FOLDER = os.path.join(os.getcwd(), "Data")

# Raw data
RAW_DATA_ZIP_FILENAME = "RawData.zip"
MIT_DATA_FOLDER = os.path.join(DATA_FOLDER, "Test Data")

# Generated files names: # TODO: beautify all names
TAPS_INPUT = os.path.join(DATA_FOLDER, "TAPS.csv")
USERS_INPUT = os.path.join(DATA_FOLDER, "USERS.csv")
KAGGLE_DATA_ARTICLE_METHOD1_FEATURES = os.path.join(DATA_FOLDER, "final.csv")
MIT_USERS_INPUT = os.path.join(DATA_FOLDER, "MIT_USERS.csv")
MIT_TAPS_INPUT = os.path.join(DATA_FOLDER, "MIT_TAPS.csv")
MIT_FINAL_DATASET = os.path.join(DATA_FOLDER, "MIT_final.csv")
MIT_NQI_FEATURES = os.path.join(DATA_FOLDER, "MIT_NQI_FEATURES.csv")
MIT_NQI_FEATURES_SIDES_PARTITIONS = os.path.join(DATA_FOLDER, "MIT_NQI_FEATURES_SIDES_PARTITIONS.csv")
KAGGLE_TAPS_INPUT = os.path.join(DATA_FOLDER, "KAGGLE_TAPS.csv")
KAGGLE_USERS_INPUT = os.path.join(DATA_FOLDER, "KAGGLE_USERS.csv")
KAGGLE_NQI_FEATURES = os.path.join(DATA_FOLDER, "KAGGLE_NQI_FEATURES.csv")

# Others
TAPS_THRESHOLD = 2000
