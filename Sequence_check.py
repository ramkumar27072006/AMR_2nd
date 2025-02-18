from Bio import Entrez, SeqIO

# List of gene sequence IDs to download
gene_ids =gene_ids = [
    "NM_001200025", "NM_000546", "NM_001354670", "NM_001301794", "NM_002046",
    "NM_001369055", "NM_001142334", "NM_015658", "NM_007294", "NM_001636",
    "NM_004333", "NM_001256799", "NM_000518", "NM_001042351", "NM_021972",
    "NM_000497", "NM_000314", "NM_003002", "NM_001127511", "NM_004985",
    "NM_000251", "NM_001033", "NM_001354698", "NM_004415", "NM_004448",
    "NM_000151", "NM_002524", "NM_004360", "NM_001317736", "NM_000487",
    "NM_000492", "NM_001172", "NM_002422", "NM_004938", "NM_005732",
    "NM_000059", "NM_002755", "NM_002521", "NM_005343", "NM_001282550",
    "NM_003998", "NM_005228", "NM_001001130", "NM_020975", "NM_005251",
    "NM_001172421", "NM_000350", "NM_005633", "NM_001127208", "NM_001364928",
    "NM_006218", "NM_020529", "NM_001025195", "NM_003019", "NM_000546"
]

# Set your email (required by NCBI Entrez for identification)
Entrez.email = "ramayanam001@gmail.com.com"  # Replace with your email

# Directory to save downloaded sequences
output_dir = r"D:\2ND SEM\BIO\review"

# Download gene sequences in FASTA format
for gene_id in gene_ids:
    try:
        # Fetch the gene sequence from NCBI
        with Entrez.efetch(db="nucleotide", id=gene_id, rettype="fasta", retmode="text") as handle:
            fasta_data = handle.read()
        
        # Save the sequence to a file
        output_file = f"{output_dir}\\{gene_id}.fasta"
        with open(output_file, "w") as file:
            file.write(fasta_data)
        
        print(f"Downloaded and saved: {output_file}")
    except Exception as e:
        print(f"Failed to download {gene_id}: {e}")