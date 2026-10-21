"""ScreenSight entry point. No network calls, downloads, or admin elevation."""
import argparse
from screensight.engine import bundled_model


def self_test() -> int:
    from screensight.core import Letterbox, parse_yolo_people
    import numpy as np
    # Deterministic smoke test for the detection decoder.
    result = np.zeros((1, 84, 2), dtype=np.float32)
    result[0, 0:4, 0] = [320, 320, 100, 160]
    result[0, 4, 0] = .95
    boxes = parse_yolo_people(result, Letterbox(1, 0, 0, 640, 640), .5)
    assert len(boxes) == 1 and boxes[0].score > .9, boxes
    model = bundled_model()
    if not model.exists():
        raise RuntimeError(f'Bundled model not found: {model}')
    from screensight.detector import PersonDetector
    engine = PersonDetector(model)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    detections, elapsed = engine.detect(img, .9)
    print('PASS: decode, ONNX load, inference', engine.provider, f'{elapsed:.1f} ms', len(detections))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='ScreenSight local vision overlay')
    parser.add_argument('--self-test', action='store_true', help='Offline inference smoke test')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    from screensight.gui import MainWindow
    MainWindow().run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
