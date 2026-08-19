"""Explicit opt-in remote pyannote Precision-2 diarization provider."""

from __future__ import annotations

from voxdelta.credentials import Credentials
from voxdelta.domain.models import ProviderProvenance
from voxdelta.providers.pyannote_diarization import PipelineFactory, _PyannoteAdapter

PRECISION_MODEL_ID = "pyannote/speaker-diarization-precision-2"
PYANNOTE_DATA_RETENTION_URL = "https://docs.pyannote.ai/data-retention"


class PyannotePrecisionProvider(_PyannoteAdapter):
    """Use Precision-2 only when this remote provider is selected explicitly."""

    def __init__(
        self,
        credentials: Credentials,
        *,
        retention_policy_url: str = PYANNOTE_DATA_RETENTION_URL,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        super().__init__(
            credentials,
            model_reference=PRECISION_MODEL_ID,
            provenance=ProviderProvenance(
                name="pyannote",
                model="speaker-diarization-precision-2",
                remote=True,
                transmits=("audio",),
                retention_policy_url=retention_policy_url,
            ),
            credential_name="pyannoteai",
            pipeline_factory=pipeline_factory,
        )
