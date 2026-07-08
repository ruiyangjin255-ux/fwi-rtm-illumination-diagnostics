from scripts.run_protocol_v12_spectrogram_dinov2_confirmation import build_optimizer_with_registration_report


def test_decoder_parameters_are_registered_in_optimizer() -> None:
    import torch

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.decoder = torch.nn.Linear(4, 2)

    model = Model()
    _, report = build_optimizer_with_registration_report(model, learning_rate=1e-3)
    assert report["decoder_parameters"] > 0
    assert report["decoder_optimizer_parameters"] == report["decoder_parameters"]
    assert report["decoder_fully_registered"] is True
    assert report["optimizer_parameters"] == report["trainable_parameters"]


def test_registration_report_handles_inactive_lazy_decoder_branch() -> None:
    import torch

    class LazyDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vector = torch.nn.LazyLinear(2)
            self.map = torch.nn.LazyConv2d(2, 1)

        def forward(self, x):
            return self.vector(x)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = LazyDecoder()

    model = Model(); model.decoder(torch.ones(1, 4))
    _, report = build_optimizer_with_registration_report(model, learning_rate=1e-3)
    assert report["decoder_fully_registered"] is True
    assert report["uninitialized_decoder_parameter_objects"] == 2
