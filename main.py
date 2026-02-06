from data_cleaning import preprocess_flight_data
import pandas as pd 





df = pd.read_csv("./data/Flight_Price_Dataset_of_Bangladesh.csv")









preprocess_flight_data(df)