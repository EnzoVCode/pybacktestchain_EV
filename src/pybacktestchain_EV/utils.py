import random

# List of famous guitar players, bike names, and gear brands
guitar_players = [
    "Hendrix", "Clapton", "Page", "SRV", "Gilmour", 
    "VanHalen", "Satriani", "Slash", "Santana", "Blackmore", 
    "Hammett", "Iommi", "Smith", "Beck", "Young"
]

bike_names = [
    "FatBoy", "RoadKing", "Scout", "Monster", 
    "Bonneville", "R1", "Ninja", "nineT", 
    "Hayabusa", "GoldWing", "FatBob", "ElectraGlide",
]

gear_brands = [
    "Fender", "Gibson", "Marshall", "Mesa Boogie", "Ibanez", "PRS", 
    "Orange", "Gretsch", "Dunlop", "Boss"
]

# Function to generate a random name
def generate_random_name():
    player = random.choice(guitar_players)
    bike = random.choice(bike_names)
    gear = random.choice(gear_brands)
    return f"{player}-{bike}-{gear}"

