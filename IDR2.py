import os
import csv
import numpy as np
from collections import defaultdict

idr_folder = "./idr_results/final_idrs"
filtered_folder = "./idr_results/filtered_idr_results"
os.makedirs(filtered_folder, exist_ok=True)

no_idr_file = os.path.join(filtered_folder, "noID-PSP.txt")
summary_file = os.path.join(filtered_folder, "retained_idrs_summary.csv")

def refine_idrs(res_ids):
    if not res_ids:
        return [], []

    min_id, max_id = min(res_ids), max(res_ids)
    length = max_id - min_id + 1
    seq = np.zeros(length, dtype=int)
    for res_id in res_ids:
        seq[res_id - min_id] = 1

    # Remove short IDR segments (≤3)
    i = 0
    while i < length:
        if seq[i] == 1:
            start = i
            while i < length and seq[i] == 1:
                i += 1
            if i - start <= 3:
                seq[start:i] = 0
        else:
            i += 1

    # Convert a short ordered region (≤ 10) between two long IDRs
    i = 0
    while i < length:
        if seq[i] == 0:
            start = i
            while i < length and seq[i] == 0:
                i += 1
            end = i
            left_len = right_len = 0
            l = start - 1
            while l >= 0 and seq[l] == 1:
                left_len += 1
                l -= 1
            r = end
            while r < length and seq[r] == 1:
                right_len += 1
                r += 1
            if (end - start <= 10) and (left_len >= 20) and (right_len >= 20):
                seq[start:end] = 1
        else:
            i += 1

    
    # Extract all final IDR segments (of any length, as long as they remain after processing)
    final_ids = []
    retained_segments = []
    i = 0
    while i < length:
        if seq[i] == 1:
            start = i
            while i < length and seq[i] == 1:
                i += 1
            segment = list(range(start + min_id, i + min_id))
            final_ids.extend(segment)
            retained_segments.append((segment[0], segment[-1], len(segment)))
        else:
            i += 1

    return final_ids, retained_segments

with open(summary_file, "w", newline='') as summary_f, open(no_idr_file, "w") as no_idr_log:
    summary_writer = csv.writer(summary_f)
    summary_writer.writerow(["Protein ID", "Chain ID", "Start", "End", "Length"])

    for filename in os.listdir(idr_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(idr_folder, filename)
            pdb_id = os.path.splitext(filename)[0]
            output_path = os.path.join(filtered_folder, filename)

            chain_idrs = defaultdict(list)
            with open(input_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    chain_id = row["Chain ID"]
                    res_id = int(row["Residue ID"])
                    chain_idrs[chain_id].append(res_id)

            final_results = []
            all_segments = []

            for chain_id, res_ids in chain_idrs.items():
                refined_ids, segments = refine_idrs(res_ids)
                for rid in refined_ids:
                    final_results.append((chain_id, rid))
                for (start, end, length) in segments:
                    summary_writer.writerow([pdb_id, chain_id, start, end, length])

            if final_results:
                with open(output_path, "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Chain ID", "Residue ID"])
                    for chain_id, res_id in sorted(final_results):
                        writer.writerow([chain_id, res_id])
                print(f" Preserve IDR: {filename}")
            else:
                no_idr_log.write(pdb_id + "\n")
                print(f" No IDR preserved: {filename} → classified as noID-PSP")
