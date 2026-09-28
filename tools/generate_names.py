import json
import os

# Define the naval-themed components
boat_naming_template = {
    "fishing_boats": {
        "first_word": [
            "Pacific", "Ocean", "Northern", "Stormy", "Golden", 
            "Silver", "Wild", "Mystic", "Deep", "Coastal", 
            "Lucky", "Morning", "Harbour", "Island", "Salty", 
            "Arctic", "Glacial", "Brave", "Wandering", "Emerald"
        ],
        "last_word": [
            "Mariner", "Seafarer", "Gull", "Wave", "Crest", 
            "Star", "Horizon", "Current", "Drifter", "Voyager", 
            "Dawn", "Tide", "Breeze", "Pearl", "Grace", 
            "Fortune", "Runner", "Seas", "Quest", "Legacy"
        ]
    },
    "patrol_boats": {
        "first_word": [
            "Guardian", "Sentinel", "Vigilant", "Valiant", "Defiant", 
            "Steadfast", "Relentless", "Stalwart", "Fearless", "Dauntless", 
            "Apex", "Iron", "Storm", "Swift", "Bold", 
            "Tracker", "Hunter", "Active", "Pacific", "Coast"
        ],
        "last_word": [
            "Protector", "Interceptor", "Enforcer", "Ranger", "Defender", 
            "Vanguard", "Shield", "Falcon", "Hawk", "Eagle", 
            "Saber", "Blade", "Watchman", "Warden", "Patriot", 
            "Pursuit", "Scout", "Keeper", "Knight", "Victory"
        ]
    }
}

# Ensure the output directory exists safely
os.makedirs("datasets", exist_ok=True)
output_path = "datasets/boat_names.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(boat_naming_template, f, indent=2)

print(f"📦 Successfully generated unified name dictionary text asset at: {output_path}")
