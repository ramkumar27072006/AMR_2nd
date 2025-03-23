import requests

def get_protein_info_pdb(protein_id):
    # URL to fetch protein data from PDB API (e.g., structure data in JSON format)
    url = f"https://data.rcsb.org/rest/v1/core/entry/{protein_id}"
    
    # Make a request to the PDB API
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract protein-related information
        name = data.get("struct", {}).get("title", "N/A")
        organism = data.get("struct", {}).get("organism", "N/A")
        
        # Handle 'exptl' field which is a list
        exptl_data = data.get("exptl", [])
        structure_method = exptl_data[0]["method"] if exptl_data else "N/A"
        
        # Fetch the structure file URL (PDB format)
        structure_url = f"https://files.rcsb.org/download/{protein_id}.pdb"
        
        return {
            "Protein Name": name,
            "Organism": organism,
            "Structure Method": structure_method,
            "Structure URL": structure_url
        }
    else:
        return None

# Example: Using PDB API to fetch details for a protein (e.g., P53 human protein structure)
protein_id = "1TUP"  # Replace with a valid PDB ID (e.g., p53 structure)
protein_info = get_protein_info_pdb(protein_id)

if protein_info:
    print(f"Protein Name: {protein_info['Protein Name']}")
    print(f"Organism: {protein_info['Organism']}")
    print(f"Structure Method: {protein_info['Structure Method']}")
    print(f"Structure URL: {protein_info['Structure URL']}")
else:
    print("Protein information not found.")
