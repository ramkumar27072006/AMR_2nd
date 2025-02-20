Fasta_File_Merge

Get-Content "file_path(1)", "file_path(2)", "file_path(3)" | Set-Content "output_file_path"
Example:
Get-Content "D:\2nd_SEM\fasta_file\sequence (1).fasta", "D:\2nd_SEM\fasta_file\sequence (2).fasta", "D:\2nd_SEM\fasta_file\sequence (3).fasta" | Set-Content "D:\2ND SEM\delete\combined_sequences.fasta"


Fasta file to csv

# Define input FASTA file and output CSV file
$inputFile = "D:\2ND SEM\file_path"
$outputFile = "D:\2ND SEM\output_file_path"

# Initialize an array to store sequences
$sequences = @()
$header = ""
$sequence = ""

# Read the FASTA file line by line
Get-Content $inputFile | ForEach-Object {
    if ($_ -match "^>") {
        # If a new header is found, save the previous sequence
        if ($header -ne "") {
            $sequences += [PSCustomObject]@{Header=$header; Sequence=$sequence}
        }
        # Start a new sequence
        $header = $_ -replace "^>", ""  # Remove '>' from header
        $sequence = ""
    } else {
        # Append sequence data
        $sequence += $_
    }
}

# Add the last sequence to the array
if ($header -ne "") {
    $sequences += [PSCustomObject]@{Header=$header; Sequence=$sequence}
}

# Export to CSV
$sequences | Export-Csv -Path $outputFile -NoTypeInformation

Write-Host "FASTA file has been converted to CSV: $outputFile"
