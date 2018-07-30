from localConstants import * #todo: remove this file
import os

# the folder containing "Tappy Data" and "Archived Users" folders with the Kaggle data,
# and to which the output files will be saved also
# DATA_FOLDER = os.getcwd() TODO: uncomment

# Generated files names:
MIT_DATA_FOLDER = os.path.join(DATA_FOLDER, "Test Data") #TODO: change name to MIT, change other filename as well to be nicer
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
TAPS_THRESHOLD = 2000 # changes to 500 for test data

