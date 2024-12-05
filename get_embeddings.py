import os

import pandas as pd
import pickle
import torch
from tqdm import tqdm

import whisper

torch.cuda.empty_cache()
model = whisper.load_model("large-v2")

audio_dir = "../PROCESS-V1"

metadata_file = os.path.join(audio_dir, "dem-info.csv")
metadata = pd.read_csv(metadata_file)

extractions = []

for folder in tqdm(os.listdir(audio_dir)):
    if folder.startswith("Process-rec-"):
        extraction = {}
        extraction["metadata"] = metadata[metadata["Record-ID"] == folder].iloc[0].to_dict()
        for test in ["CTD", "PFT", "SFT"]:
            audio_fp = os.path.join(audio_dir, folder, f"{folder}__{test}.wav")
            embedding_fp = os.path.join(emb_dir, f"{folder}__{test}.pt")

            audio = whisper.load_audio(audio_fp)
            with torch.no_grad():
                embeddings = model.embed(audio)

            print(embeddings.shape)
            extraction[test] = embeddings

        extractions.append(extraction)

with open("embeddings.pkl", "wb") as f:
    pickle.dump(extractions, f)
