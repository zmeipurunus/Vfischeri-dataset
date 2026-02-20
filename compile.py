import re
import csv
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "vibrio_genbank.txt")
output_file = os.path.join(script_dir, "vibrio_proteins_dataset.csv")

proteins = []

with open(input_file, "r") as f:
	lines = f.readlines()

current_translation = None
current_protein_id = None
current_locus_tag = None
collecting_translation = False

for line in lines:
	line = line.rstrip()

	# Capture protein_id
	protein_match = re.search(r'/protein_id="([^"]+)"', line)
	if protein_match:
		current_protein_id = protein_match.group(1)

	# Capture locus_tag
	locus_match = re.search(r'/locus_tag="([^"]+)"', line)
	if locus_match:
		current_locus_tag = locus_match.group(1)

	# Start translation block
	if '/translation="' in line:
		collecting_translation = True
		current_translation = ""
		line_part = line.split('/translation="')[1]
		
		if line_part.endswith('"'):
			current_translation += line_part[:-1]
			collecting_translation = False
		else:
			current_translation += line_part
		continue

	# Continue translation block
	if collecting_translation:
		if line.endswith('"'):
			current_translation += line.strip().rstrip('"')
			collecting_translation = False
		else:
			current_translation += line.strip()
		continue

	# When translation ends, store it
	if not collecting_translation and current_translation:
		# Clean sequence (only amino acids)
		sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', current_translation)

		if len(sequence) > 0:
			proteins.append({
				"protein_id": current_protein_id,
				"locus_tag": current_locus_tag,
				"sequence": sequence,
				"length": len(sequence)
			})

		current_translation = None

# Write dataset
with open(output_file, "w", newline="") as csvfile:
	fieldnames = ["protein_id", "locus_tag", "sequence", "length"]
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for protein in proteins:
		writer.writerow(protein)

print(f"Extracted {len(proteins)} protein sequences.")
