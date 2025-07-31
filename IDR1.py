import os
import csv
from Bio import PDB
from Bio.PDB import DSSP

# === Enter the PDB folder ===
pdb_folder="./outputs"

# === 设置输出目录 ===
plddt_folder = "./idr_results/plddt_scores"
idrs_folder = "./idr_results/final_idrs"
residue_info_folder = "./idr_results/residue_info"
os.makedirs(plddt_folder, exist_ok=True)
os.makedirs(idrs_folder, exist_ok=True)
os.makedirs(residue_info_folder, exist_ok=True)

parser = PDB.PDBParser(QUIET=True)

for filename in os.listdir(pdb_folder):
    if filename.endswith(".pdb"):
        pdb_path = os.path.join(pdb_folder, filename)
        pdb_id = os.path.splitext(filename)[0]

        try:
            structure = parser.get_structure(pdb_id, pdb_path)

            # Extracting the pLDDT score (B-factor)
            pLDDT_scores = {}
            residue_names = {}
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if PDB.is_aa(residue):
                            try:
                                pLDDT_scores[(chain.id, residue.get_id()[1])] = residue["CA"].get_bfactor()
                                residue_names[(chain.id, residue.get_id()[1])] = residue.get_resname()
                            except KeyError:
                                continue  # Some residues may not have CA atoms

            # DSSP Notes
            dssp = DSSP(structure[0], pdb_path)
            secondary_structure = {}
            for key in dssp.keys():
                chain_id, res_id = key[0], key[1][1]
                ss = dssp[key][2]
                secondary_structure[(chain_id, res_id)] = ss

            # Preliminary disordered region: pLDDT < 50
            idrs = [res_id for res_id, pLDDT in pLDDT_scores.items() if pLDDT < 50]

            # Ordered structure definition: H, G, I, B, E
            ordered_regions = [
                res_id for res_id, ss in secondary_structure.items()
                if ss in ['H', 'G', 'I', 'B', 'E']
            ]

            # Final IDRs = pLDDT low + not in ordered structure
            final_idrs = [res_id for res_id in idrs if res_id not in ordered_regions]

            # === Saving pLDDT Scores ===
            with open(os.path.join(plddt_folder, f"{pdb_id}_plddt.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Chain ID", "Residue ID", "pLDDT Score"])
                for (chain_id, res_id), pLDDT in sorted(pLDDT_scores.items()):
                    writer.writerow([chain_id, res_id, pLDDT])

            # === Save the final unordered area ===
            with open(os.path.join(idrs_folder, f"{pdb_id}_idrs.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Chain ID", "Residue ID"])
                for (chain_id, res_id) in sorted(final_idrs):
                    writer.writerow([chain_id, res_id])

            # === Save complete residue information ===
            with open(os.path.join(residue_info_folder, f"{pdb_id}_residues.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Chain ID", "Residue ID", "Residue Name", "pLDDT Score", "Secondary Structure"])
                all_residues = sorted(pLDDT_scores.keys())
                for (chain_id, res_id) in all_residues:
                    resname = residue_names.get((chain_id, res_id), "UNK")
                    plddt = pLDDT_scores.get((chain_id, res_id), "")
                    ss = secondary_structure.get((chain_id, res_id), "C")  # When missing, it is recorded as C (coil)
                    writer.writerow([chain_id, res_id, resname, plddt, ss])

            print(f"{pdb_id} analysis completed.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
