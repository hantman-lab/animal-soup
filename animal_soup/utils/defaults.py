from pathlib import Path
import os

pretrained_path = Path(os.path.abspath(__file__)).parent.parent.joinpath("pretrained_models")

FLOW_GEN_MODEL_PATHS = {
    "table": {
        "TinyMotionNet3D": pretrained_path.joinpath(
            "table", "flow_generator", "flow_generator",
        ).with_name("TinyMotionNet3D.ckpt"),
        "TinyMotionNet": pretrained_path.joinpath(
            "table", "flow_generator", "flow_generator",
        ).with_name("TinyMotionNet.ckpt"),
        "MotionNet": pretrained_path.joinpath(
            "table", "flow_generator", "flow_generator"
        ).with_name(
            "MotionNet.ckpt"
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
        ).with_name("hidden_two_stream_fast.ckpt"),
        "medium": pretrained_path.joinpath(
            "table", "feature_extractor", "feature_extractor"
        ).with_name("hidden_two_stream_medium.ckpt"),
        "slow": pretrained_path.joinpath(
            "table", "feature_extractor", "feature_extractor"
        ).with_name("hidden_two_stream_slow.ckpt")
    },
    "pez": {

    }
}


