import os

import pandas as pd
import pickle
import torch
from tqdm import trange

import whisper

torch.cuda.empty_cache()
model = whisper.load_model("large-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"

audio_dir = "../PROCESS-V1"

metadata_file = os.path.join(audio_dir, "dem-info.csv")
metadata = pd.read_csv(metadata_file)

extractions = []

for n in range(6, 16):
    for i in trange(n*10 + 1, min((n+1) * 10 + 1, 158)):
        folder = f"Process-rec-{i:03d}"
        if folder.startswith("Process-rec-"):
            extraction = {}
            extraction["metadata"] = metadata[metadata["Record-ID"] == folder].iloc[0].to_dict()
            for test in ["CTD", "PFT", "SFT"]:
                audio_fp = os.path.join(audio_dir, folder, f"{folder}__{test}.wav")

                audio = whisper.load_audio(audio_fp)
                audio = torch.from_numpy(audio).to(device)
                with torch.no_grad():
                    embeddings = model.embed(audio)

                print(embeddings.shape)
                extraction[test] = embeddings

            extractions.append(extraction)

    with open(f"embeddings{n}.pkl", "wb") as f:
        pickle.dump(extractions, f)
