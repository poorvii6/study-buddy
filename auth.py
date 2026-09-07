import hashlib
import os
import secrets


# Turn a plain password into a safe, scrambled hash we can store.
# We NEVER store the actual password — only this one-way hash.
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return salt.hex() + ":" + dk.hex()


# Check a login attempt against the stored hash.
def verify_password(password: str, stored: str) -> bool:
    salt_hex, dk_hex = stored.split(":")
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return secrets.compare_digest(dk.hex(), dk_hex)


# A random login token, handed to the browser after a successful login.
def new_token() -> str:
    return secrets.token_hex(16)
