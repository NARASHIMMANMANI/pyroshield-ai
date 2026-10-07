import ee

try:
    ee.Initialize(project="erudite-river-502705-n5")
except:
    ee.Initialize()


def get_nightlights(latitude, longitude, date):

    point = ee.Geometry.Point([longitude, latitude])

    start = ee.Date(date).advance(-30, "day")
    end = ee.Date(date).advance(30, "day")

    collection = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterBounds(point)
        .filterDate(start, end)
        .sort("system:time_start")
    )

    image = collection.first()

    if image is None:
        return {
            "night_light": None
        }

    value = image.select("avg_rad").reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=500,
        maxPixels=1e9
    ).getInfo()

    return {
        "night_light": value.get("avg_rad")
    }