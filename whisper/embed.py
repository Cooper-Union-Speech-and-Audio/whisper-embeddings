from typing import Union

import numpy as np
import torch

from .audio import log_mel_spectrogram, N_SAMPLES

# @torch.no_grad
def embed(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
):
    mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)
    
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    trimmed_length = (mel.shape[1] // model.dims.n_audio_ctx) * model.dims.n_audio_ctx
    mel = mel[:, :trimmed_length]
    mel = mel.reshape(-1, model.dims.n_mels, model.dims.n_audio_ctx)
    print(mel.shape)

    # return model.embed_audio(mel)
