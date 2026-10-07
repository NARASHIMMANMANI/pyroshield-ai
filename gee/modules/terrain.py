import ee

try:
    ee.Initialize(project="erudite-river-502705-n5")
except:
    ee.Initialize()


def get_terrain(latitude, longitude):

    point = ee.Geometry.Point([longitude, latitude])

    dem = ee.Image("USGS/SRTMGL1_003")

    terrain = ee.Terrain.products(dem)

    elevation = dem.select("elevation")

    slope = terrain.select("slope")

    aspect = terrain.select("aspect")

    image = elevation.addBands(slope).addBands(aspect)

    values = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=30
    ).getInfo()

    return values