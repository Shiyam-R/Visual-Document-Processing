"""
app/exceptions.py
─────────────────────────────────────────────────────────────────────────────
Custom exception hierarchy — same pattern as the churn project. Each
exception carries its own HTTP status code so main.py's handler can
return a consistent structured JSON error.
"""

from typing import Optional


class DocumentAPIError(Exception):
    def __init__(self, message: str, status_code: int = 500, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail or message


class ArtifactLoadError(DocumentAPIError):
    def __init__(self, artifact: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to load artifact: {artifact}.",
            status_code=503,
            detail=f"Artifact '{artifact}' could not be loaded. Reason: {reason}.",
        )


class ModelNotLoadedError(DocumentAPIError):
    def __init__(self) -> None:
        super().__init__(
            message="Model is not loaded.",
            status_code=503,
            detail="The LayoutLM model was not found or failed to load during startup.",
        )


class InvalidImageError(DocumentAPIError):
    def __init__(self, reason: str) -> None:
        super().__init__(
            message="Uploaded file is not a valid image.",
            status_code=422,
            detail=reason,
        )


class OCRError(DocumentAPIError):
    def __init__(self, reason: str) -> None:
        super().__init__(
            message="OCR failed to detect any text in the image.",
            status_code=422,
            detail=reason,
        )


class PredictionError(DocumentAPIError):
    def __init__(self, reason: str) -> None:
        super().__init__(
            message="Model inference failed.",
            status_code=500,
            detail=reason,
        )


class ExtractionNotFoundError(DocumentAPIError):
    def __init__(self, extraction_id: int) -> None:
        super().__init__(
            message=f"Extraction {extraction_id} not found.",
            status_code=404,
            detail=f"No extraction record exists with id={extraction_id}.",
        )
