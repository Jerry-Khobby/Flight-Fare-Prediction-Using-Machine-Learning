from data_cleaning import preprocess_flight_data
import pandas as pd 
from data_exploration import perform_eda_kpis



df = pd.read_csv("./data/cleaned_flight_data.csv")


perform_eda_kpis(df)