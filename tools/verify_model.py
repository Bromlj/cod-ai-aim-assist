"""Hard fail if upstream ONNX model differs from the reviewed artifact hash."""
import hashlib
from pathlib import Path

source = Path('models/person-yolo11n.onnx')
expected = '7d8fd1717d9d5bbab6986cd134afb620649c7a394303d55b1e09fc00804cc5c1'
assert source.is_file(), f'Not found: {source}'
sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert sha == expected, f'Model checksum mismatch: {sha}'
print('ONNX model verified:', sha)
