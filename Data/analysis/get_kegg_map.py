import requests
import json

def fetch_kegg_species_data():
    """
    Fetches the list of organisms from the KEGG API and saves it to a JSON file.
    The JSON file will map KEGG Organism IDs to their corresponding species names.
    """
    api_url = "http://rest.kegg.jp/list/organism"
    species_to_id_map = {}
    
    print("Fetching data from KEGG API...")
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        
        for line in response.text.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 3:
                kegg_id = parts[1]
                species_name = parts[2]
                species_to_id_map[kegg_id] = species_name

        output_filename = 'kegg_species_map.json'
        
        with open(output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(species_to_id_map, json_file, ensure_ascii=False, indent=4)
            
        print(f"\nSuccessfully extracted {len(species_to_id_map)} species.")
        print(f"The data has been saved to '{output_filename}'")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from the KEGG API: {e}")

# Run the function
fetch_kegg_species_data()