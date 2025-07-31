import os
import torch
import esm
import time
from Bio import SeqIO
import biotite.structure.io as bsio
from multiprocessing import Process, Queue, set_start_method

def run_on_gpu(gpu_id, queue):
    torch.cuda.set_device(gpu_id)
    model = esm.pretrained.esmfold_v1()
    model.set_chunk_size(64)
    model = model.eval().cuda()

    while not queue.empty():
        record = queue.get()
        seq_id = record.id
        sequence = str(record.seq).strip()

        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            # save
            pdb_file = f"outputs/{seq_id}.pdb"
            with open(pdb_file, "w") as f:
                f.write(output)

            struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
            print(f"[GPU {gpu_id}]  {seq_id} pLDDT: {struct.b_factor.mean():.2f}")

        except Exception as e:
            print(f"[GPU {gpu_id}]  {seq_id} failed：{e}")

def main():
    fasta_path = "./data.fasta"
    os.makedirs("outputs", exist_ok=True)
    records = list(SeqIO.parse(fasta_path, "fasta"))

    queue = Queue()
    for record in records:
        queue.put(record)

    num_gpus = torch.cuda.device_count()
    print(f" Use {num_gpus} GPUs")

    processes = []
    for i in range(num_gpus):
        p = Process(target=run_on_gpu, args=(i, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All protein predictions completed")

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    main()


