import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.model import build_model


def test_model_creation():

    model = build_model(6)

    assert model is not None
    assert len(model.layers) == 3


def test_model_output():

    model = build_model(6)

    assert model.output_shape[-1] == 1
