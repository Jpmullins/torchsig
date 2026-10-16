"""Band-realistic synthetic spectrum: a named band, the emitters expected in it, and a streaming receiver.

Where :class:`torchsig.datasets.TorchSigIterableDataset` makes a scene from a list of modulation classes at
arbitrary frequencies, this package starts from a real band and asks what is actually transmitting there.
A spectrum allocation table, channel plans and deployment-density figures give the population; per-technology
activity models give the timing; the TorchSig signal builders render the waveforms; and a receiver model
delivers the result as the sample format a real radio hands over.

    from torchsig.spectrum import BandStream, Receiver, named_band

    stream = BandStream(named_band("ism_2450"), Receiver(tune_hz=2441.75e6, decimation=2), seed=7)
    for chunk in stream.take(100):
        ...  # chunk.sc16 is [n, 2] int16; chunk.bursts is the truth

The pieces, in the order they compose:

* :mod:`torchsig.spectrum.catalog` -- what is allocated where (:class:`BandPlan`, :class:`Technology`).
* :mod:`torchsig.spectrum.population` -- who is out there, on which channel, how far, how loud.
* :mod:`torchsig.spectrum.activity` -- when each one transmits (:class:`Schedule`, :class:`Burst`).
* :mod:`torchsig.spectrum.waveforms` -- rendering one burst through a TorchSig builder.
* :mod:`torchsig.spectrum.receiver` -- the X310/UBX-160 front end and its artifacts.
* :mod:`torchsig.spectrum.stream` -- :class:`BandStream`, endless and seekable.
"""

from torchsig.spectrum.activity import Burst, Schedule
from torchsig.spectrum.catalog import BandPlan, Technology, load_catalog, named_band
from torchsig.spectrum.population import Emitter, Environment, populate
from torchsig.spectrum.receiver import Receiver
from torchsig.spectrum.stream import BandStream, Chunk
from torchsig.spectrum.waveforms import WaveformSpec, render_burst

__all__ = [
    "BandPlan",
    "BandStream",
    "Burst",
    "Chunk",
    "Emitter",
    "Environment",
    "Receiver",
    "Schedule",
    "Technology",
    "WaveformSpec",
    "load_catalog",
    "named_band",
    "populate",
    "render_burst",
]
