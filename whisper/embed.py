from typing import Union

import numpy as np
import torch

from .audio import log_mel_spectrogram, N_SAMPLES, N_FRAMES

# @torch.no_grad
def embed(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
):
    mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)

    trimmed_length = (mel.shape[1] // N_FRAMES) * N_FRAMES
    mel = mel[:, :trimmed_length]
    mel = mel.reshape(
        trimmed_length // N_FRAMES,
        model.dims.n_mels,
        N_FRAMES,
    )

    embedding = model.embed_audio(mel)
    return embedding.view(-1, 384)
