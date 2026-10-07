import ee
import math

# Initialize Earth Engine
try:
    ee.Initialize(project="erudite-river-502705-n5")
except:
    ee.Initialize()


def get_weather(latitude, longitude, date):

    point = ee.Geometry.Point([longitude, latitude])

    start = ee.Date(date)
    end = start.advance(1, "day")

    image = (
        ee.ImageCollection("ECMWF/ERA5/DAILY")
        .filterDate(start, end)
        .first()
    )

    if image is None:
        return None

    data = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=10000
    ).getInfo()

    if data is None:
        return None

    u = data.get("u_component_of_wind_10m")
    v = data.get("v_component_of_wind_10m")

    wind_speed = None

    if u is not None and v is not None:
        wind_speed = math.sqrt(u*u + v*v)

    return {

    "temperature":
        data.get("mean_2m_air_temperature") - 273.15,

    "min_temperature":
        data.get("minimum_2m_air_temperature") - 273.15,

    "max_temperature":
        data.get("maximum_2m_air_temperature") - 273.15,

    "dewpoint":
        data.get("dewpoint_2m_temperature") - 273.15,

    "precipitation":
        data.get("total_precipitation"),

    "surface_pressure":
        data.get("surface_pressure"),

    "sea_level_pressure":
        data.get("mean_sea_level_pressure"),

    "u_wind":
        u,

    "v_wind":
        v,

    "wind_speed":
        wind_speed
}