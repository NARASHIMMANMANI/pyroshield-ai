from modules.nightlights import get_nightlights

night = get_nightlights(
    latitude=11.0168,
    longitude=76.9558,
    date="2020-07-09"
)

print(night)