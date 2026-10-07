import ee

try:
    ee.Initialize(project="erudite-river-502705-n5")
except:
    ee.Initialize()


def get_vegetation(latitude, longitude, date):

    point = ee.Geometry.Point([longitude, latitude])

    start = ee.Date(date).advance(-15, "day")
    end = ee.Date(date).advance(15, "day")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    image = collection.first()

    if image is None:
        return None

    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")

    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")

    evi = image.expression(
        "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
        {
            "NIR": image.select("B8"),
            "RED": image.select("B4"),
            "BLUE": image.select("B2")
        }
    ).rename("EVI")

    stacked = ndvi.addBands(evi).addBands(ndwi).addBands(nbr)

    values = stacked.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=10
    ).getInfo()

    return values