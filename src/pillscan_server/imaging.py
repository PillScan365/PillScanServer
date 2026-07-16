import warnings
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter

from anyio import to_thread
from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from pillscan_server.config import Settings
from pillscan_server.errors import ImageValidationError
from pillscan_server.protocols import PreparedImage

SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
READ_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True, slots=True)
class PreparedUpload:
    image: PreparedImage
    upload_read_ms: float
    image_normalization_ms: float


async def prepare_upload(
    upload: UploadFile,
    settings: Settings,
) -> PreparedUpload:
    if upload.content_type not in SUPPORTED_MEDIA_TYPES:
        raise ImageValidationError(
            f"image must be JPEG, PNG, or WebP; got {upload.content_type or 'unknown'}"
        )

    read_started_at = perf_counter()
    raw = await _read_limited(upload, settings.max_upload_bytes)
    upload_read_ms = _elapsed_ms(read_started_at)

    normalization_started_at = perf_counter()
    image = await to_thread.run_sync(
        _normalize_image,
        raw,
        settings.max_image_pixels,
        settings.max_image_dimension,
    )
    return PreparedUpload(
        image=image,
        upload_read_ms=upload_read_ms,
        image_normalization_ms=_elapsed_ms(normalization_started_at),
    )


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(READ_CHUNK_SIZE):
        total += len(chunk)
        if total > limit:
            raise ImageValidationError(f"image exceeds the {limit // (1024 * 1024)} MB limit")
        chunks.append(chunk)

    if total == 0:
        raise ImageValidationError("image is empty")
    return b"".join(chunks)


def _normalize_image(
    raw: bytes,
    max_pixels: int,
    max_dimension: int,
) -> PreparedImage:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ImageValidationError("image has an unsupported format")
                if source.width * source.height > max_pixels:
                    raise ImageValidationError(
                        f"image exceeds the {max_pixels:,}-pixel safety limit"
                    )

                transposed = ImageOps.exif_transpose(source)
                rgb = _onto_white_background(transposed)
                rgb.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

                output = BytesIO()
                rgb.save(output, format="JPEG", quality=95, subsampling=0, optimize=True)
                return PreparedImage(
                    media_type="image/jpeg",
                    data=output.getvalue(),
                    width=rgb.width,
                    height=rgb.height,
                )
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageValidationError("image is too large to process safely") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise ImageValidationError("upload is not a valid image") from None


def _onto_white_background(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")
