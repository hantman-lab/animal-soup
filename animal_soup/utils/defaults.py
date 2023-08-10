from pathlib import Path
import os

pretrained_path = Path(os.path.abspath(__file__)).parent.parent.joinpath("pretrained_models")

FLOW_GEN_MODEL_PATHS = {
    "table": {
        "TinyMotionNet3D": pretrained_path.joinpath(
            "table", "flow_generator", "flow_generator",
        ).with_name("TinyMotionNet3D.pt"),
        "TinyMotionNet": pretrained_path.joinpath(
            "table", "flow_generator", "flow_generator",
        ).with_name("TinyMotionNet.pt"),
        "MotionNet": pretrained_path.joinpath(
            "table", "flow_generator", "flow_generator"
        ).with_name(
            "MotionNet.pt"
        )
    },
    "pez": {

    }

}

RESNET34_3D_PATH = pretrained_path.joinpath("misc", "misc").with_name("resnet34_3d.pth")

FEATURE_EXTRACTOR_MODEL_PATHS = {
    "table": {
        "fast": pretrained_path.joinpath(
            "table", "feature_extractor", "feature_extractor"
        ).with_name("hidden_two_stream_fast.pt"),
        "medium": pretrained_path.joinpath(
            "table", "feature_extractor", "feature_extractor"
        ).with_name("hidden_two_stream_medium.pt"),
        "slow": pretrained_path.joinpath(
            "table", "feature_extractor", "feature_extractor"
        ).with_name("hidden_two_stream_slow.pt")
    },
    "pez": {

    }
}

SEQUENCE_MODEL_PATHS = {
    "table":
        {
            "fast": pretrained_path.joinpath(
                "table", "sequence_model", "sequence_model"
            ).with_name('tgmj_fast.pt'),
            "medium": pretrained_path.joinpath(
                "table", "sequence_model", "sequence_model"
            ).with_name('tgmj_medium.pt'),
            "slow": pretrained_path.joinpath(
                "table", "sequence_model", "sequence_model"
            ).with_name('tgmj_slow.pt')
        },
    "pez": {

        }
}


def download_pretrained():
    """Download the pre-trained default models from zenodo."""
