# Define the templates for tank and artillery units
import json

tank_template = {
    "path": [],
    "name": None,
    "category": "tank",
    "role": "combat",
    "ranged_strength_base": 0,
    "range_radius_base": 0,
    "combat_strength_base": 85,
    "hp": 100,
    "mp": 4,
    "mp_base": 4,
    "image_path": "assets/units/tank.png"
}

artillery_template = {
    "path": [],
    "name": None,
    "category": "artillery",
    "role": "ranged",
    "ranged_strength_base": 90,
    "range_radius_base": 2,
    "combat_strength_base": 70,
    "hp": 100,
    "mp": 3,
    "mp_base": 3,
    "image_path": "assets/units/artillery.png"
}

tank_names = [
    "Rampage Ranger",
    "Shield Shredder",
    "Blaze Brute",
    "Titan Tremor",
    "Iron Invader",
    "Grit Grinder",
    "Steel Sledge",
    "Armor Avenger",
    "Thunder Tank",
    "Mighty Mauler",
    "Fury Fighter",
    "Metal Marauder",
    "Tough Tumbler",
    "Valor Vanguard",
    "Force Frenzy",
    "Dread Destroyer",
    "Warrior Wrecker",
    "Impact Igniter",
    "Power Pulverizer",
    "Battle Bulldozer",
    "Guardian Goliath",
    "Siege Sentinel",
    "Vigil Vindicator",
    "Colossus Crusher",
    "Juggernaut Jolt",
    "Clash Cavalier",
    "Annihilator Alpha",
    "Bravo Berserker",
    "Delta Devastator",
    "Echo Eradicator"
]

artillery_names = [
    "Strife Striker",
    "Blast Blazer",
    "Cannon Commander",
    "Missile Master",
    "Explosive Enforcer",
    "Howitzer Hero",
    "Mortar Maverick",
    "Payload Paladin",
    "Shell Shooter",
    "Ordnance Officer",
    "Bombardier Bravo",
    "Siege Specialist",
    "Ricochet Ranger",
    "Trajectory Titan",
    "Detonator Delta",
    "Fireball Fiend",
    "Rocketeer Rebel",
    "Caliber Captain",
    "Grenade Guardian",
    "Pyro Punisher",
    "Impact Invoker",
    "Volley Vanguard",
    "Fusillade Fury",
    "Scattergun Sage",
    "Thunder Thumper",
    "Ballistic Baron",
    "Snipe Sniper",
    "Eagle Eye",
    "Rainmaker",
    "Boom Bringer"
]


# Function to generate units based on user input with the correct input format
def generate_units_from_input():
    tank_names_iter = iter(tank_names[::-1])
    artillery_names_iter = iter(artillery_names[::-1])

    # Ask the user for lists of coordinates for tanks and artilleries
    tank_coordinates_input = input("Enter the list of coordinates for tanks (e.g. (15, 4) (12, 5)): ")

    # Convert the input strings to lists of tuples
    try:
        tank_coordinates = [tuple(map(int, coord.strip("()").split(","))) for coord in
                            tank_coordinates_input.split(") (")]
    except ValueError:
        print("Invalid input format. Please try again.")
        return

    # Initialize an empty list to hold the new units
    new_units = []
    # Generate tank units
    for r, c in tank_coordinates:
        new_unit = tank_template.copy()
        new_unit['r'] = r
        new_unit['c'] = c
        new_unit['name'] = next(tank_names_iter)
        new_units.append(new_unit)

    print(json.dumps(new_units))

    new_units = []
    artillery_coordinates_input = input("Enter the list of coordinates for artilleries (e.g. (11, 6) (16, 5)): ")
    try:
        artillery_coordinates = [tuple(map(int, coord.strip("()").split(","))) for coord in
                                 artillery_coordinates_input.split(") (")]
    except ValueError:
        print("Invalid input format. Please try again.")
        return

    # Generate artillery units
    for r, c in artillery_coordinates:
        new_unit = artillery_template.copy()
        new_unit['r'] = r
        new_unit['c'] = c
        new_unit['name'] = next(artillery_names_iter)
        new_units.append(new_unit)

    print(json.dumps(new_units))


if __name__ == '__main__':
    generate_units_from_input()
