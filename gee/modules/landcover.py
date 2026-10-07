import ee

try:
    ee.Initialize(project="erudite-river-502705-n5")
except:
    ee.Initialize()


def get_landcover(latitude, longitude):

    point = ee.Geometry.Point([longitude, latitude])

    landcover = ee.Image("ESA/WorldCover/v200/2021")

    value = landcover.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=10
    ).getInfo()

    landcover_classes = {
        10: "Tree Cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / Sparse Vegetation",
        70: "Snow and Ice",
        80: "Permanent Water",
        90: "Herbaceous Wetland",
        95: "Mangroves",
        100: "Moss and Lichen"
    }

    code = value.get("Map")

    return {
        "landcover_code": code,
        "landcover_name": landcover_classes.get(code, "Unknown")
    }