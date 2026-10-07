import ee

ee.Initialize(project="erudite-river-502705-n5")

collection = ee.ImageCollection("WorldPop/GP/100m/pop")

print("Number of Images:")
print(collection.size().getInfo())

image = collection.first()

print("\nBands:")
print(image.bandNames().getInfo())

print("\nProperties:")
print(image.propertyNames().getInfo())