from modules.vegetation import get_vegetation

veg = get_vegetation(
    latitude=11.0168,
    longitude=76.9558,
    date="2020-07-09"
)

print(veg)