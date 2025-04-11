import gzip
import csv
from Bio import SeqIO

# Input and output paths
fasta_gz_path = r"C:\Users\R RAMKUMAR\Downloads\uniprotkb_non_virulence_protein_2025_03_15.fasta.gz"
output_csv_path = r"C:\Users\R RAMKUMAR\Downloads\uniprotkb_non_virulence_protein_2025_03_15.fasta\virulence_proteins.csv"

# Open and parse the gzipped FASTA
with gzip.open(fasta_gz_path, 'rt') as handle, open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Entry ID', 'Description', 'Sequence'])  # CSV header

    for record in SeqIO.parse(handle, 'fasta'):
        entry_id = record.id
        description = record.description
        sequence = str(record.seq)
        csvwriter.writerow([entry_id, description, sequence])
