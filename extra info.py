import requests
from xml.etree import ElementTree

# Gene ID obtained from the previous search result
gene_id = "4298"  # Test with a random Gene ID, e.g., BRCA1 (Human gene ID)

# Use esummary to get detailed information about the gene
summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id={gene_id}&retmode=xml"

summary_response = requests.get(summary_url)

# Check if the request was successful
if summary_response.status_code == 200:
    # Parse the summary XML
    summary_tree = ElementTree.fromstring(summary_response.content)
    
    # Extract the DocumentSummary element for the gene
    document_summary = summary_tree.find(".//DocumentSummary")
    
    if document_summary is not None:
        print("Gene Information:")
        
        # Extract specific elements and print them
        name = document_summary.find("Name").text if document_summary.find("Name") is not None else "N/A"
        description = document_summary.find("Description").text if document_summary.find("Description") is not None else "N/A"
        genetic_source = document_summary.find("GeneticSource").text if document_summary.find("GeneticSource") is not None else "N/A"
        
        # Extract and handle multiple organisms (if there are multiple)
        organisms = document_summary.findall(".//Organism/ScientificName")
        organism_names = [organism.text for organism in organisms if organism.text is not None]

        print(f"Gene Name: {name}")
        print(f"Description: {description}")
        print(f"Genetic Source: {genetic_source}")
        print(f"Organisms: {', '.join(organism_names) if organism_names else 'N/A'}")
    else:
        print("No gene information found.")
else:
    print("Error retrieving detailed information.")
