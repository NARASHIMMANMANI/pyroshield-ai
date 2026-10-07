from modules.weather import get_weather

weather = get_weather(

    latitude=11.0168,
    longitude=76.9558,
    date="2020-07-09"

)

print(weather)