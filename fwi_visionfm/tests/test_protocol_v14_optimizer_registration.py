from __future__ import annotations


def test_protocol_v14_optimizer_registration_covers_bridge_film_and_decoder():
    from scripts.run_protocol_v14_geometry_aware_trace_bridge import build_geometry_bridge_optimizer_report
    from fwi_visionfm.models.geometry_aware_trace_bridge import GeometryAwareTraceBridge
    from fwi_visionfm.models.protocol_v11_common_decoder import build_protocol_v11_decoder

    bridge = GeometryAwareTraceBridge(token_dim=16, use_multiscale_context=True)
    decoder = build_protocol_v11_decoder(output_shape=(70, 70), base_channels=16, vmin=1500.0, vmax=4500.0).module
    report = build_geometry_bridge_optimizer_report(bridge=bridge, decoder=decoder, learning_rate=1.0e-3)
    assert report["optimizer_parameters"] == report["trainable_parameters"]
    assert report["film_parameters"] > 0
    assert report["decoder_fully_registered"] is True
