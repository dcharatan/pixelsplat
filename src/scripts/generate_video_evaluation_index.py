import json
from pathlib import Path

INDEX_INPUT = Path("assets/evaluation_index_re10k.json")
INDEX_OUTPUT = Path("assets/evaluation_index_re10k_video.json")

if __name__ == "__main__":
    with INDEX_INPUT.open("r") as f:
        index_input = json.load(f)

    index_output = {}
    for scene, scene_index_input in index_input.items():
        # Handle scenes for which there's no index.
        if scene_index_input is None:
            index_output[scene] = None
            continue

        # Add all intermediate frames as target frames.
        a, b = scene_index_input["context"]
        index_output[scene] = {
            "context": [a, b],
            "target": list(range(a, b + 1)),
        }

    with INDEX_OUTPUT.open("w") as f:
        json.dump(index_output, f)
