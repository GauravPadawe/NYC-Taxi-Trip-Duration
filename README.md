# NYC Taxi Trip Duration

# DOMAIN: Transportation
# NYC Taxi Trip Duration
# By Gaurav Padawe - 24th October, 2018

The dataset is based on the 2016 NYC Yellow Cab trip record data made available in Big Query on Google Cloud Platform. The data was originally published by the NYC Taxi and Limousine Commission (TLC). The data was sampled and cleaned for the purposes of this playground. Based on individual trip attributes, should predict the duration of each trip.

## Source:

NYC Taxi and Limousine Commission (TLC) : http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml

Kaggle : https://www.kaggle.com/c/nyc-taxi-trip-duration/data

Download : https://drive.google.com/open?id=1OyOC9y2x4uyT7drXJBOEZ2yRBktiQB8H

## Details:

File descriptions ● train.csv - the dataset (contains 1458644 trip records)

## Data fields

● id - a unique identifier for each trip

● vendor_id - a code indicating the provider associated with the trip record

● pickup_datetime - date and time when the meter was engaged

● dropoff_datetime - date and time when the meter was disengaged

● passenger_count - the number of passengers in the vehicle (driver entered value)

● pickup_longitude - the longitude where the meter was engaged

● pickup_latitude - the latitude where the meter was engaged

● dropoff_longitude - the longitude where the meter was disengaged

● dropoff_latitude - the latitude where the meter was disengaged

● store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip

● trip_duration - duration of the trip in seconds

## Objective:
## Build a model that predicts the total trip duration of taxi trips in New York City.
