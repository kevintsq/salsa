import torch

from models.audio.external.MobileNetV3 import get_model as get_mobile_net, AugmentScatteringSTFT
from models.audio.external.dymn.model import get_model as get_dymn
from models.audio.external.MobileNetV3 import AugmentMelSTFT


class Wrapper(torch.nn.Module):

    def __init__(self, mel, model):
        super().__init__()
        self.mel = mel
        self.model = model

    def forward(self, x, **kwargs):
        mel = self.mel(x)
        out = self.model(mel[:, None])[:, None, None, :]
        return out


def get_efficientat(model_name='mn40_as_ext', freqm=48, timem=192, scatter=False, **kwargs):
    # get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
    if "mn40_as_ext" == model_name:
        model = get_mobile_net(width_mult=4.0, pretrained_name=model_name)
        dim = 3840
    elif "dymn20_as(4)" == model_name:
        model = get_dymn(width_mult=2.0, pretrained_name=model_name)
        dim = 1920
    else:
        raise ValueError(f"Model {model_name} not found. Available models: mn40_as_ext, dymn20_as")

    # print(model.mel)  # Extracts mel spectrogram from raw waveforms.

    if scatter:
        mel = AugmentScatteringSTFT()
    else:
        mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=freqm, timem=timem)

    wrapper = Wrapper(mel, model)

    return wrapper, dim


if __name__ == '__main__':

    from data.datasets.audioset import audioset, get_audioset
    from sacred import Experiment

    ex = Experiment('test_as', ingredients=[audioset])


    @ex.main
    def run_test():

        model = get_efficientat('mn40_as_ext', return_sequence=True)[0]

        aus = get_audioset("evaluation")
        aus.set_fixed_length(10).cache_audios()

        predicted = []
        true = []
        model.eval()
        for a in aus:
            with torch.no_grad():
                predicted.append(model(torch.from_numpy(a['audio'])[None, :]).detach().numpy())
                true.append(a['target'])

        import numpy as np
        # predicted = np.stack(predicted)
        # true = np.stack(true)
        from sklearn import metrics
        metrics.average_precision_score(np.stack(true), np.stack(predicted)[:, 0, :], average=None)
        print(predicted)


    ex.run()
