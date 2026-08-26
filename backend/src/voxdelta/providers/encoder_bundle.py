"""Verify an immutable local encoder bundle before any model is allocated from it.

The emotion2vec encoder normally arrives through a package manager's cache directory.
That directory is not a product artifact: it is mutable, it accumulates partial downloads
and editor droppings, its weights are a symlink into a second cache, and nothing about it
is attested. Handing that path to a model loader means trusting whatever happens to be in
it at the moment of the load.

So the replay evaluator never receives a cache path. It receives a *bundle*: a directory
holding only the four files the loader actually reads, every one of them a real regular
file with a recorded size and digest, described by a manifest that is itself covered by a
checksum file. Verification happens before the path is handed to the loader, so a bundle
that has been edited, truncated, extended, symlinked, or half-written is rejected while it
is still only a path — never after weights have been mapped into memory. The manifest is
the publisher's commit marker, so a directory that has been reserved but not finished
cannot be accepted here: without ``ENCODER.json`` there is nothing to verify against.

The four names are fixed by the encoder's own ``configuration.json``: ``init_param``
(``model.pt``), the tokenizer's ``token_list`` (``tokens.txt``), ``config`` (``config.yaml``),
and that configuration file itself. Documentation, images, sample audio, ``.gitattributes``,
``.DS_Store``, and any ``*.incomplete`` download are deliberately absent — a bundle
containing them is malformed, not merely untidy.

**Internal consistency is not identity.** A manifest and a checksum file prove only that a
bundle agrees with itself, and anyone who can write the directory can rewrite all three
together and still claim to be ``iic/emotion2vec_plus_large@v2.0.5``. The trusted content
is therefore pinned in this module: the exact digest and byte count of each of the four
files, and the canonical tree digest over them. Verification compares against that pin, so
a self-consistent bundle carrying different weights is rejected — and rejected while it is
still a path, before any loader has been handed it. The pin is not a parameter: the public
verifier takes only a path, so no caller can relax it.

Nothing in a bundle is derived from evaluation data. No holdout, validation audio,
transcript, item identity, or probability ever enters one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voxdelta.evaluation.manifest import read_trusted_regular_file
from voxdelta.providers._attested_tree import (
    canonical_sha256,
    digest_and_size,
    is_positive_int,
    is_sha256,
    load_strict_json,
    parse_checksums,
    resolved_artifact_root,
    safe_relative,
    tree_files,
)

ENCODER_BUNDLE_KIND = "local-encoder-bundle"
ENCODER_BUNDLE_SCHEMA_VERSION = "1"
MANIFEST_NAME = "ENCODER.json"
CHECKSUMS_NAME = "SHA256SUMS"

EMOTION2VEC_ENCODER_ID = "iic/emotion2vec_plus_large"
EMOTION2VEC_ENCODER_REVISION = "v2.0.5"
INIT_PARAM_NAME = "model.pt"

# The trusted content of iic/emotion2vec_plus_large@v2.0.5, by digest and exact size.
# These are the anchor: everything else in a bundle is only self-description.
PINNED_PAYLOADS: dict[str, tuple[str, int]] = {
    "config.yaml": (
        "f4fa0eb82cc78bfebb43c56d68791afb01788085a18897d20999af7bc45d51d3",
        5552,
    ),
    "configuration.json": (
        "8b6a745213025c7d4565f9f074cfef1b5cd5ef76e38e2a8f7f8a3db271e735b2",
        343,
    ),
    INIT_PARAM_NAME: (
        "be501a01f26fcdc7663a062dff86af839afbaef7c4de32f5e42d7e1ad2784da4",
        1945790254,
    ),
    "tokens.txt": (
        "dd99b40d17b0a78483480f02f24027f4761d65bdc7a866aacf5a8b68b5d3a564",
        90,
    ),
}
PINNED_BUNDLE_TREE_SHA256 = "a22e467cadca68fa996efc4f827b6ac18966785e45549bbdff0e99f45d0c481e"

# Exactly what the local loader reads, and nothing else.
REQUIRED_FILES: tuple[str, ...] = tuple(sorted(PINNED_PAYLOADS))

_RESERVED_NAMES = frozenset({MANIFEST_NAME, CHECKSUMS_NAME})
_REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "encoder_id",
        "encoder_revision",
        "init_param",
        "payloads",
        "bundle_tree_sha256",
    }
)
_PAYLOAD_KEYS = frozenset({"path", "sha256", "bytes"})


class EncoderBundleError(ValueError):
    """Fixed, path-free rejection of an untrusted or unrecognized encoder bundle."""

    def __init__(self) -> None:
        super().__init__("invalid_encoder_bundle")


@dataclass(frozen=True, slots=True)
class VerifiedEncoderBundle:
    """A local encoder directory proven safe to hand to the loader."""

    path: Path
    encoder_id: str
    encoder_revision: str
    init_param_sha256: str
    bundle_tree_sha256: str


def _fail() -> EncoderBundleError:
    return EncoderBundleError()


def encoder_bundle_manifest(payloads: list[dict[str, object]]) -> dict[str, object]:
    """Build the canonical manifest document for one bundle's payload records."""

    return {
        "schema_version": ENCODER_BUNDLE_SCHEMA_VERSION,
        "artifact_kind": ENCODER_BUNDLE_KIND,
        "encoder_id": EMOTION2VEC_ENCODER_ID,
        "encoder_revision": EMOTION2VEC_ENCODER_REVISION,
        "init_param": INIT_PARAM_NAME,
        "payloads": payloads,
        "bundle_tree_sha256": canonical_sha256(payloads),
    }


def _manifest_payloads(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    payloads = manifest["payloads"]
    if not isinstance(payloads, list) or not payloads:
        raise _fail()
    records: list[dict[str, object]] = []
    for payload in payloads:
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
            raise _fail()
        try:
            path = safe_relative(payload["path"], _RESERVED_NAMES)
        except ValueError:
            raise _fail() from None
        if not is_sha256(payload["sha256"]) or not is_positive_int(payload["bytes"]):
            raise _fail()
        records.append({"path": path, "sha256": payload["sha256"], "bytes": payload["bytes"]})
    paths = [str(record["path"]) for record in records]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise _fail()
    if canonical_sha256(records) != manifest["bundle_tree_sha256"]:
        raise _fail()
    return {str(record["path"]): record for record in records}


def pinned_payload_records() -> list[dict[str, object]]:
    """The pinned payload records, in the canonical order the manifest must use."""

    return [
        {"path": name, "sha256": PINNED_PAYLOADS[name][0], "bytes": PINNED_PAYLOADS[name][1]}
        for name in sorted(PINNED_PAYLOADS)
    ]


def verify_encoder_bundle(path: str | Path) -> VerifiedEncoderBundle:
    """Verify one local encoder bundle against the pinned encoder identity.

    Takes a path and nothing else: the pinned digests are deliberately not parameters, so
    there is no argument by which a caller can accept a different encoder. Raises
    ``EncoderBundleError`` for every rejection, with no host path in the message.
    """

    try:
        root = resolved_artifact_root(path)
        checksums = parse_checksums(
            read_trusted_regular_file(root / CHECKSUMS_NAME),
            manifest_name=MANIFEST_NAME,
            reserved=_RESERVED_NAMES,
        )
        manifest_raw = read_trusted_regular_file(root / MANIFEST_NAME)
        if hashlib.sha256(manifest_raw).hexdigest() != checksums[MANIFEST_NAME]:
            raise _fail()

        manifest = load_strict_json(manifest_raw)
        if not isinstance(manifest, dict) or set(manifest) != _REQUIRED_MANIFEST_KEYS:
            raise _fail()
        if (
            manifest["schema_version"] != ENCODER_BUNDLE_SCHEMA_VERSION
            or manifest["artifact_kind"] != ENCODER_BUNDLE_KIND
            or manifest["encoder_id"] != EMOTION2VEC_ENCODER_ID
            or manifest["encoder_revision"] != EMOTION2VEC_ENCODER_REVISION
            or manifest["init_param"] != INIT_PARAM_NAME
            or manifest["bundle_tree_sha256"] != PINNED_BUNDLE_TREE_SHA256
        ):
            raise _fail()

        attested = _manifest_payloads(manifest)
        # Exactly the loader's file set: nothing missing, nothing extra, nothing stray.
        if set(attested) != set(REQUIRED_FILES):
            raise _fail()
        if set(checksums) != set(attested) | {MANIFEST_NAME}:
            raise _fail()
        if tree_files(root) != set(attested) | _RESERVED_NAMES:
            raise _fail()

        # The anchor. A re-signed bundle agrees with itself but not with this.
        for name, (expected_digest, expected_bytes) in PINNED_PAYLOADS.items():
            record = attested[name]
            if record["sha256"] != expected_digest or record["bytes"] != expected_bytes:
                raise _fail()

        for relative, record in sorted(attested.items()):
            digest, size = digest_and_size(root / relative)
            if (
                digest != record["sha256"]
                or digest != checksums[relative]
                or size != record["bytes"]
            ):
                raise _fail()

        return VerifiedEncoderBundle(
            path=root,
            encoder_id=EMOTION2VEC_ENCODER_ID,
            encoder_revision=EMOTION2VEC_ENCODER_REVISION,
            init_param_sha256=PINNED_PAYLOADS[INIT_PARAM_NAME][0],
            bundle_tree_sha256=PINNED_BUNDLE_TREE_SHA256,
        )
    except EncoderBundleError:
        raise
    except Exception:
        raise _fail() from None


__all__ = [
    "CHECKSUMS_NAME",
    "EMOTION2VEC_ENCODER_ID",
    "EMOTION2VEC_ENCODER_REVISION",
    "ENCODER_BUNDLE_KIND",
    "ENCODER_BUNDLE_SCHEMA_VERSION",
    "INIT_PARAM_NAME",
    "MANIFEST_NAME",
    "PINNED_BUNDLE_TREE_SHA256",
    "PINNED_PAYLOADS",
    "REQUIRED_FILES",
    "EncoderBundleError",
    "VerifiedEncoderBundle",
    "encoder_bundle_manifest",
    "pinned_payload_records",
    "verify_encoder_bundle",
]
