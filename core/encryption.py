from __future__ import annotations

"""
backend/core/encryption.py

Kerala Flood Relief Coordination Agent
Secure field-level encryption using AES-256-GCM.

Dependency:
    pip install cryptography

Environment:
    DATA_ENCRYPTION_KEYS='{"v1":"BASE64_KEY"}'
    DATA_ENCRYPTION_ACTIVE_KEY_ID=v1

Generate a key:
    python -c "import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"

Never commit encryption keys to Git.
"""


import base64
import json
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Any, Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


logger = logging.getLogger(__name__)

NONCE_SIZE = 12
KEY_SIZE = 32
ENCRYPTION_VERSION = 1
ALGORITHM = "AES-256-GCM"


class EncryptionError(Exception):
    """Base exception for encryption operations."""


class EncryptionConfigurationError(EncryptionError):
    """Invalid or missing encryption configuration."""


class DecryptionError(EncryptionError):
    """Ciphertext could not be authenticated or decrypted."""


class UnknownKeyError(EncryptionError):
    """The requested encryption key is unavailable."""


def encode_base64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(
            value.encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, UnicodeError) as exc:
        raise DecryptionError(
            "Invalid Base64 encoding"
        ) from exc


def generate_encryption_key() -> str:
    """Generate a cryptographically secure 256-bit key."""
    return encode_base64(secrets.token_bytes(KEY_SIZE))


@dataclass(frozen=True)
class EncryptionConfig:
    keys: Mapping[str, bytes]
    active_key_id: str

    @classmethod
    def from_environment(cls) -> "EncryptionConfig":
        raw_keys = os.getenv("DATA_ENCRYPTION_KEYS")
        active_id = os.getenv(
            "DATA_ENCRYPTION_ACTIVE_KEY_ID"
        )

        if not raw_keys or not active_id:
            raise EncryptionConfigurationError(
                "Encryption keys and active key ID are required"
            )

        try:
            parsed = json.loads(raw_keys)
        except json.JSONDecodeError as exc:
            raise EncryptionConfigurationError(
                "DATA_ENCRYPTION_KEYS must be valid JSON"
            ) from exc

        if not isinstance(parsed, dict) or not parsed:
            raise EncryptionConfigurationError(
                "Encryption keys must be a nonempty object"
            )

        keys: dict[str, bytes] = {}

        for key_id, encoded_key in parsed.items():
            if (
                not isinstance(key_id, str)
                or not key_id
                or not isinstance(encoded_key, str)
            ):
                raise EncryptionConfigurationError(
                    "Invalid key configuration"
                )

            try:
                key = decode_base64(encoded_key)
            except DecryptionError as exc:
                raise EncryptionConfigurationError(
                    f"Invalid encoding for key {key_id}"
                ) from exc

            if len(key) != KEY_SIZE:
                raise EncryptionConfigurationError(
                    f"Key {key_id} must contain 32 bytes"
                )

            keys[key_id] = key

        if active_id not in keys:
            raise EncryptionConfigurationError(
                "Active encryption key is unavailable"
            )

        return cls(
            keys=keys,
            active_key_id=active_id,
        )


class EncryptionService:
    """
    Authenticated encryption service.

    Uses AES-256-GCM with a fresh random nonce
    for every encryption operation.

    The key identifier is authenticated as
    additional associated data.
    """

    def __init__(self, config: EncryptionConfig):
        if config.active_key_id not in config.keys:
            raise EncryptionConfigurationError(
                "Active key not found"
            )

        for key in config.keys.values():
            if len(key) != KEY_SIZE:
                raise EncryptionConfigurationError(
                    "All keys must be 256-bit"
                )

        self.config = config

    @classmethod
    def from_environment(cls) -> "EncryptionService":
        return cls(EncryptionConfig.from_environment())

    def encrypt_bytes(
        self,
        plaintext: bytes,
        *,
        context: str,
    ) -> str:
        if not context:
            raise ValueError(
                "Encryption context must not be empty"
            )

        key_id = self.config.active_key_id
        key = self.config.keys[key_id]

        nonce = secrets.token_bytes(NONCE_SIZE)

        aad = self._associated_data(
            key_id=key_id,
            context=context,
        )

        ciphertext = AESGCM(key).encrypt(
            nonce,
            plaintext,
            aad,
        )

        envelope = {
            "version": ENCRYPTION_VERSION,
            "algorithm": ALGORITHM,
            "key_id": key_id,
            "nonce": encode_base64(nonce),
            "ciphertext": encode_base64(ciphertext),
        }

        return json.dumps(
            envelope,
            separators=(",", ":"),
        )

    def decrypt_bytes(
        self,
        encrypted: str,
        *,
        context: str,
    ) -> bytes:
        if not context:
            raise ValueError(
                "Encryption context must not be empty"
            )

        try:
            envelope = json.loads(encrypted)
        except (TypeError, ValueError) as exc:
            raise DecryptionError(
                "Invalid encrypted envelope"
            ) from exc

        if not isinstance(envelope, dict):
            raise DecryptionError(
                "Encrypted envelope must be an object"
            )

        if (
            type(envelope.get("version")) is not int
            or envelope["version"] != ENCRYPTION_VERSION
            or envelope.get("algorithm") != ALGORITHM
        ):
            raise DecryptionError(
                "Unsupported encryption format"
            )

        key_id = envelope.get("key_id")

        if not isinstance(key_id, str):
            raise DecryptionError("Invalid key identifier")

        key = self.config.keys.get(key_id)

        if key is None:
            raise UnknownKeyError(
                "Encryption key is unavailable"
            )

        nonce_value = envelope.get("nonce")
        ciphertext_value = envelope.get("ciphertext")

        if (
            not isinstance(nonce_value, str)
            or not isinstance(ciphertext_value, str)
        ):
            raise DecryptionError(
                "Invalid ciphertext fields"
            )

        nonce = decode_base64(nonce_value)
        ciphertext = decode_base64(ciphertext_value)

        if len(nonce) != NONCE_SIZE:
            raise DecryptionError(
                "Invalid encryption nonce"
            )

        if len(ciphertext) < 16:
            raise DecryptionError(
                "Invalid ciphertext length"
            )

        aad = self._associated_data(
            key_id=key_id,
            context=context,
        )

        try:
            return AESGCM(key).decrypt(
                nonce,
                ciphertext,
                aad,
            )
        except InvalidTag as exc:
            raise DecryptionError(
                "Ciphertext authentication failed"
            ) from exc

    def encrypt_text(
        self,
        plaintext: str,
        *,
        context: str,
    ) -> str:
        return self.encrypt_bytes(
            plaintext.encode("utf-8"),
            context=context,
        )

    def decrypt_text(
        self,
        encrypted: str,
        *,
        context: str,
    ) -> str:
        plaintext = self.decrypt_bytes(
            encrypted,
            context=context,
        )

        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecryptionError(
                "Decrypted data is not valid UTF-8"
            ) from exc

    def encrypt_json(
        self,
        data: Any,
        *,
        context: str,
    ) -> str:
        serialized = json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return self.encrypt_text(
            serialized,
            context=context,
        )

    def decrypt_json(
        self,
        encrypted: str,
        *,
        context: str,
    ) -> Any:
        plaintext = self.decrypt_text(
            encrypted,
            context=context,
        )

        try:
            return json.loads(plaintext)
        except json.JSONDecodeError as exc:
            raise DecryptionError(
                "Decrypted JSON is invalid"
            ) from exc

    def rotate_ciphertext(
        self,
        encrypted: str,
        *,
        context: str,
    ) -> str:
        """
        Re-encrypt data using the current active key.

        Old keys must remain configured until
        existing ciphertext is migrated.
        """

        plaintext = self.decrypt_bytes(
            encrypted,
            context=context,
        )

        return self.encrypt_bytes(
            plaintext,
            context=context,
        )

    def get_key_id(self, encrypted: str) -> str:
        try:
            envelope = json.loads(encrypted)

            if not isinstance(envelope, dict):
                raise ValueError("Invalid envelope")

            key_id = envelope["key_id"]

            if not isinstance(key_id, str):
                raise ValueError("Invalid key identifier")

            return key_id

        except (TypeError, ValueError, KeyError) as exc:
            raise DecryptionError(
                "Cannot read encryption key ID"
            ) from exc

    @staticmethod
    def _associated_data(
        *,
        key_id: str,
        context: str,
    ) -> bytes:
        return json.dumps(
            {
                "version": ENCRYPTION_VERSION,
                "algorithm": ALGORITHM,
                "key_id": key_id,
                "context": context,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def create_encryption_service() -> EncryptionService:
    return EncryptionService.from_environment()


def encrypt_sensitive_data(
    data: Any,
    *,
    context: str,
    service: EncryptionService,
) -> str:
    return service.encrypt_json(
        data,
        context=context,
    )


def decrypt_sensitive_data(
    encrypted: str,
    *,
    context: str,
    service: EncryptionService,
) -> Any:
    return service.decrypt_json(
        encrypted,
        context=context,
    )
