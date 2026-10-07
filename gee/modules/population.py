import ee

try:
    ee.Initialize(project="erudite-river-502705-n5")
except:
    ee.Initialize()


def get_population(latitude, longitude):

    point = ee.Geometry.Point([longitude, latitude])

    image = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("year", 2020))
        .filterBounds(point)
        .first()
    )

    if image is None:
        return {
            "population_density": None
        }

    value = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=100,
        maxPixels=1e9
    ).getInfo()

    return {
        "population_density": value.get("population")
    }