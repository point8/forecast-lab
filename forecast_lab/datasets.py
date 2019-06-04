import pandas

def read_usa_temperature(data_path="../data/usa-avg-temp-monthly.csv"):

    def fahrenheit_to_celsius(f):
        c = (f - 32) * 5/9
        return c

    usa_temp = pandas.read_csv(
        data_path,
        dtype={"Date": "str"}
    )
    # get parsable date format
    usa_temp["Date"] = usa_temp["Date"].apply(lambda s: f"{s[:4]}-{s[4:]}")
    # convert units
    usa_temp["Value"] = usa_temp["Value"].apply(fahrenheit_to_celsius)
    usa_temp["Anomaly"] = usa_temp["Anomaly"].apply(fahrenheit_to_celsius)
    # datetime index
    usa_temp["Date"] = pandas.to_datetime(usa_temp["Date"])
    usa_temp = usa_temp.set_index("Date")
    return usa_temp

def read_chicago_taxi_trips_daily(data_path="../data/taxi_trips_daily.csv"):
    taxi_trips = pandas.read_csv(
        data_path,
        sep=";",
        parse_dates=["Date"]
    )
    taxi_trips = taxi_trips.set_index("Date")
    taxi_trips["Trips"].freq = "d"
    return taxi_trips
