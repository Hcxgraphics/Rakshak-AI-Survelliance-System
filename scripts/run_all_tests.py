# scripts/run_all_tests.py
"""
Runs all four model tests in sequence.
Usage:
    python scripts/run_all_tests.py [--image path/to/test.jpg] [--video path/to/test.mp4]
"""
import argparse, sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path("src/deployment").resolve()))

PASS = "✅"
FAIL = "❌"
results = {}


def test_weapon():
    from ultralytics import YOLO
    model = YOLO("models/weapon/best.pt")
    dummy = np.random.randint(0, 255, (640, 640, 3), np.uint8)
    preds = model.predict(source=dummy, conf=0.01, verbose=False)
    results["weapon"] = (PASS, f"{len(preds[0].boxes)} detections on noise")


def test_violence():
    import tensorflow as tf
    model = tf.keras.models.load_model("models/violence/final_model.h5", compile=False)
    assert len(model.input_shape) in [4, 5], f"Expected 4D or 5D input, got {model.input_shape}"
    if len(model.input_shape) == 5:
        assert model.input_shape[1] == 16, "Expected 16 frames"
        clip = np.random.rand(1, 16, 64, 64, 3).astype(np.float32)
    else:
        # frame CNN fallback
        h, w = model.input_shape[1], model.input_shape[2]
        clip = np.random.rand(1, h, w, 3).astype(np.float32)
    score = float(model.predict(clip, verbose=0).flatten()[0])
    assert 0.0 <= score <= 1.0
    results["violence"] = (PASS, f"score={score:.3f}")


def test_police():
    import torch, torch.nn.functional as F, tensorflow as tf
    from torchvision import transforms
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    from PIL import Image

    backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features.eval()
    head = tf.keras.models.load_model("models/police/police_or_danger.h5", compile=False)
    assert head.output_shape[-1] == 5, f"Expected 5-class head, got {head.output_shape}"

    t = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), np.uint8))
    tensor = t(dummy_img).unsqueeze(0)
    with torch.no_grad():
        feat = F.adaptive_avg_pool2d(backbone(tensor), (1,1)).view(1, -1).cpu().numpy()
    pred = head.predict(feat, verbose=0)[0]
    results["police"] = (PASS, f"top={['NonViolence','Violence','guns','knife','police'][pred.argmax()]}")


def test_accident():
    import torch
    from accidentModel import load_accident_model
    from torchvision import transforms
    from PIL import Image

    device = torch.device("cpu")
    model  = load_accident_model(Path("models/accident/accident_model.pth"), device=device)
    t = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    dummy = t(Image.fromarray(np.random.randint(0,255,(224,224,3),np.uint8))).unsqueeze(0)
    with torch.no_grad():
        import torch.nn.functional as F
        probs = F.softmax(model(dummy)[0], dim=0).numpy()
    results["accident"] = (PASS, f"class={probs.argmax()} conf={probs.max():.3f}")


def test_full_inference(image_path=None):
    """Run the complete inference pipeline on a real or dummy frame."""
    import cv2
    from inference import detect_objects, INPUT_IMAGE

    if image_path and Path(image_path).exists():
        frame = cv2.imread(image_path)
    else:
        frame = np.random.randint(80, 180, (480, 640, 3), np.uint8)
        print("  (using random frame — pass --image for real test)")

    out = detect_objects(frame, source_type=INPUT_IMAGE)
    results["pipeline"] = (
        PASS if "image" in out else FAIL,
        f"scene={out.get('scene','?')} risk={out.get('risk_level','?')}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None)
    p.add_argument("--video", default=None)
    args = p.parse_args()

    for name, fn in [
        ("weapon",   test_weapon),
        ("violence", test_violence),
        ("police",   test_police),
        ("accident", test_accident),
        ("pipeline", lambda: test_full_inference(args.image)),
    ]:
        try:
            print(f"Running test for: {name}...")
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[name] = (FAIL, str(exc))

    print("\n═══════════ TEST RESULTS ═══════════")
    for model_name, (status, detail) in results.items():
        print(f"  {status}  {model_name:<12} {detail}")

    if all(s == PASS for s, _ in results.values()):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed — check error messages above.")
        sys.exit(1)
