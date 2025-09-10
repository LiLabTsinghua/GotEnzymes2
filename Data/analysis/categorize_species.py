import json
import csv
from io import StringIO

with open('all_species_temperatures.csv', 'r', encoding='utf-8') as f: # got from GOSHA database
    ogt_file_content = f.read()

def categorize_species_by_ogt(kegg_data_path, ogt_data_string):
    try:
        with open(kegg_data_path, 'r', encoding='utf-8') as f:
            kegg_id_to_name = json.load(f)
    except FileNotFoundError:
        print(f"errpr: KEGG'{kegg_data_path}' not found.")
        return None
    ogt_map = {}
    f = StringIO(ogt_data_string)
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) > 1 and row[0] and row[1]:
            organism_name = row[0].strip()
            try:
                ogt = float(row[1])
                ogt_map[organism_name] = ogt
            except ValueError:
                continue
    
    print(f"{len(ogt_map)} data points with OGT information loaded.")

    categorized_species = {
        "Psychrophilic (< 15°C)": [],
        "Mesophilic (15°C - 45°C)": [],
        "Thermophilic (45°C - 80°C)": [],
        "Hyperthermophilic (> 80°C)": []
    }

    matches_found = 0
    for kegg_id, full_kegg_name in kegg_id_to_name.items():
        for ogt_name, ogt_value in ogt_map.items():
            if full_kegg_name.strip().startswith(ogt_name):

                if ogt_value < 15:
                    categorized_species["Psychrophilic (< 15°C)"].append(kegg_id)
                elif 15 <= ogt_value <= 45:
                    categorized_species["Mesophilic (15°C - 45°C)"].append(kegg_id)
                elif 45 < ogt_value <= 80:
                    categorized_species["Thermophilic (45°C - 80°C)"].append(kegg_id)
                else: # > 80
                    categorized_species["Hyperthermophilic (> 80°C)"].append(kegg_id)
                matches_found += 1
                break 

    print(f"Matching completed. Successfully found OGT information for {matches_found} out of {len(kegg_id_to_name)} KEGG species.")
    return categorized_species

kegg_json_file = 'kegg_species_map.json' 
final_result = categorize_species_by_ogt(kegg_json_file, ogt_file_content)
if final_result:
    with open('categorized_species_by_ogt.json', 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

