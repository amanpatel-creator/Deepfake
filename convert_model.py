
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np

# Limit CPU threads
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "deepfake_mobilenetv3.pth"
ONNX_PATH = BASE_DIR / "models" / "deepfake_mobilenetv3.onnx"

def main():
    print("DeepShield AI - ONNX Conversion Started")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading trained PyTorch model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError("Unsupported model architecture.")

    class_map = checkpoint.get("class_map", {})

    if "real" not in class_map or "fake" not in class_map:
        raise ValueError("Invalid class map in checkpoint.")

    model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    image_size = int(checkpoint.get("image_size", 224))

    dummy_input = torch.randn(
        1, 3, image_size, image_size
    )

    print("Converting model to ONNX...")

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(ONNX_PATH),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamo=False
        )

    print("Checking ONNX model...")

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    print("Testing ONNX Runtime inference...")

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    test_input = np.random.randn(
        1, 3, image_size, image_size
    ).astype(np.float32)

    with torch.inference_mode():
        torch_output = model(
            torch.from_numpy(test_input)
        ).numpy()

    onnx_output = session.run(
        ["logits"],
        {"input": test_input}
    )[0]

    difference = np.max(
        np.abs(torch_output - onnx_output)
    )

    print("Maximum output difference:", difference)

    print("Class map:", class_map)
    print("Image size:", image_size)
    print("ONNX model saved:", ONNX_PATH)
    print("ONNX file size:", round(ONNX_PATH.stat().st_size / (1024 * 1024), 2), "MB")

    if not np.allclose(
        torch_output,
        onnx_output,
        rtol=1e-3,
        atol=1e-4
    ):
        raise RuntimeError("ONNX output does not match PyTorch output.")

    print("SUCCESS: ONNX model converted and verified!")

if __name__ == "__main__":
    main()
