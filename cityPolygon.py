import requests
import pandas as pd
import os
import time
import googlemaps
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
CSV_FILE = "city_to_city_polygon.csv"

def decode_polyline(encoded_polyline):
    """Decode Google encoded polyline to a list of lat/lng tuples."""
    return googlemaps.convert.decode_polyline(encoded_polyline)

def get_route_polygon(city_a, city_b):
    """Fetch route polyline, distance, and duration from Google Routes API."""
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "routes.polyline.encodedPolyline,routes.distanceMeters,routes.duration"
    }

    body = {
        "origin": {"address": city_a},
        "destination": {"address": city_b},
        "travelMode": "DRIVE"
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        print(f"HTTP Error {response.status_code}: {response.text}")
        return None, None, None

    data = response.json()
    if "error" in data:
        print(f"API Error: {data['error']['message']}")
        return None, None, None

    if "routes" not in data or not data["routes"]:
        print(f"No route found between {city_a} and {city_b}")
        return None, None, None

    encoded_polyline = data["routes"][0]["polyline"]["encodedPolyline"]
    decoded_polyline = decode_polyline(encoded_polyline)
    distance = data["routes"][0].get("distanceMeters", "N/A")
    duration = data["routes"][0].get("duration", "N/A")

    return decoded_polyline, distance, duration

def save_to_csv(city_a, city_b, polygon, distance, duration):
    """Save route data to CSV if it doesn't already exist."""
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        try:
            df = pd.read_csv(CSV_FILE)
        except pd.errors.EmptyDataError:
            print("Error: CSV file is empty. Creating a new file.")
            df = pd.DataFrame(columns=["CityA", "CityB", "Polygon", "Distance (meters)", "Duration (seconds)"])
    else:
        df = pd.DataFrame(columns=["CityA", "CityB", "Polygon", "Distance (meters)", "Duration (seconds)"])

    existing_entry = df[((df["CityA"] == city_a) & (df["CityB"] == city_b)) |
                        ((df["CityA"] == city_b) & (df["CityB"] == city_a))]

    if not existing_entry.empty:
        print(f"Route between {city_a} and {city_b} already exists.")
        return

    polygon_str = json.dumps(polygon)

    new_entry = pd.DataFrame([{
        "CityA": city_a,
        "CityB": city_b,
        "Polygon": polygon_str,
        "Distance (meters)": distance,
        "Duration (seconds)": duration
    }])

    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    print(f"Saved: {city_a} to {city_b} | Distance: {distance}m | Duration: {duration}s")

if __name__ == "__main__":
    while True:
        city_a = input("Enter City A (or 'exit' to quit): ").strip()
        if city_a.lower() == "exit":
            break

        city_b = input("Enter City B: ").strip()

        print(f"Fetching route from {city_a} to {city_b}...")
        polygon, distance, duration = get_route_polygon(city_a, city_b)

        if polygon:
            save_to_csv(city_a, city_b, polygon, distance, duration)
        else:
            print("No polygon data available.")
        time.sleep(1)
