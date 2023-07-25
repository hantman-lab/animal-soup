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
        "MotionNet": pretrained_path.joinpath("table", "flow_generator", "flow_generator").with_name(
            "MotionNet.ckpt"
        )
    },
    "pez": {

    }

}
