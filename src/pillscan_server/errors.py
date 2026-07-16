class PillScanError(Exception):
    status_code = 500
    code = "internal_error"
    public_message = "The request could not be completed."


class ImageValidationError(PillScanError):
    status_code = 400
    code = "invalid_image"

    def __init__(self, message: str) -> None:
        self.public_message = message
        super().__init__(message)


class VisionProviderError(PillScanError):
    status_code = 502
    code = "vision_provider_error"
    public_message = "The vision provider could not complete the analysis."


class RateLimitExceeded(PillScanError):
    status_code = 429
    code = "analysis_capacity_exceeded"
    public_message = "Analysis capacity is busy. Retry after a short delay."
