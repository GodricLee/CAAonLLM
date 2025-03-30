import hashlib
import base64

hash_bytes = hashlib.sha256(str(1).encode('utf-8')).digest()
counter_str = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')[:8]

print("counter_str:", counter_str)