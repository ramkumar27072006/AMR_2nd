import requests
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def analyze_protein(sequence):
    # Basic properties
    analysis = ProteinAnalysis(sequence)
    print(f"Length: {len(sequence)}")
    print(f"Molecular Weight: {analysis.molecular_weight():.2f}")
    print(f"Isoelectric Point: {analysis.isoelectric_point():.2f}")
    print(f"Aromaticity: {analysis.aromaticity():.3f}")
    print(f"Instability Index: {analysis.instability_index():.2f}")
    print(f"GRAVY: {analysis.gravy():.3f}")

    # Amino acid composition
    print("\nAmino Acid Composition:")
    for aa, freq in analysis.amino_acids_percent.items():
        print(f"  {aa}: {freq*100:.2f}%")

    # Secondary structure
    helix, turn, sheet = analysis.secondary_structure_fraction()
    print("\nSecondary Structure Fraction:")
    print(f"  Helix: {helix:.3f}")
    print(f"  Turn: {turn:.3f}")
    print(f"  Sheet: {sheet:.3f}")

    # Additional analyses can be added here using APIs or local tools

if _name_ == "_main_":
    sequence = input("Enter protein sequence: ").strip()
    if sequence:
        analyze_protein(sequence)
    else:
        print("Please enter a valid protein sequence.")
