import pytest
from pydantic import ValidationError

from pillscan_server.models import (
    CatalogCandidate,
    DrugIngredient,
    DrugProduct,
    DrugResolution,
    MedicationImageAnalysis,
    ProductIdentifiers,
    ResolutionSource,
    ResolutionStatus,
)


def product() -> DrugProduct:
    return DrugProduct(
        identifiers=ProductIdentifiers(
            tfda_permit_number="衛署藥製字第012345號",
            tfda_ingredient_codes=["A001234"],
            nhi_code="AC12345100",
            gtins=["04712345678901"],
        ),
        brand_name_zh="範例錠",
        brand_name_en="EXAMPLE TABLETS",
        generic_display_name="ACETAMINOPHEN 500 MG",
        ingredients=[
            DrugIngredient(
                official_name="ACETAMINOPHEN",
                normalized_generic_name="acetaminophen",
                tfda_ingredient_code="A001234",
                prescription_label=None,
                amount_description=None,
                amount="500",
                unit="MG",
            )
        ],
        dosage_form="錠劑",
        manufacturer="範例藥廠",
        applicant=None,
        indications=None,
        source_urls=[],
    )


def test_catalog_exact_has_fixed_official_product_shape() -> None:
    resolution = DrugResolution(
        status=ResolutionStatus.CATALOG_EXACT,
        source=ResolutionSource.TFDA_NHI,
        product=product(),
        candidates=[],
        catalog_version="2026-07-16",
    )

    payload = resolution.model_dump(mode="json")
    assert payload["product"]["identifiers"]["tfda_permit_number"] == "衛署藥製字第012345號"
    assert payload["product"]["ingredients"][0]["official_name"] == "ACETAMINOPHEN"
    assert payload["product"]["ingredients"][0]["amount"] == "500"


def test_catalog_exact_requires_tfda_permit_number() -> None:
    invalid_product = product().model_copy(
        update={
            "identifiers": ProductIdentifiers(
                tfda_permit_number=None,
                tfda_ingredient_codes=[],
                nhi_code="AC12345100",
                gtins=[],
            )
        }
    )

    with pytest.raises(ValidationError, match="TFDA permit number"):
        DrugResolution(
            status=ResolutionStatus.CATALOG_EXACT,
            source=ResolutionSource.TFDA_NHI,
            product=invalid_product,
            candidates=[],
            catalog_version=None,
        )


def test_catalog_candidates_require_nonempty_ranked_candidates() -> None:
    candidate = CatalogCandidate(
        product=product(),
        score=0.92,
        matching_evidence=["imprint"],
        conflicting_evidence=[],
    )
    resolution = DrugResolution(
        status=ResolutionStatus.CATALOG_CANDIDATES,
        source=ResolutionSource.TFDA,
        product=None,
        candidates=[candidate],
        catalog_version="2026-07-16",
    )

    assert resolution.product is None
    assert resolution.candidates[0].score == 0.92

    with pytest.raises(ValidationError, match="requires a queried source and candidates"):
        DrugResolution(
            status=ResolutionStatus.CATALOG_CANDIDATES,
            source=ResolutionSource.TFDA,
            product=None,
            candidates=[],
            catalog_version="2026-07-16",
        )


def test_medication_document_requires_document_type() -> None:
    with pytest.raises(ValidationError, match="requires a document type"):
        MedicationImageAnalysis.model_validate(
            {
                "subject_type": "medication_document",
                "document_type": "none",
                "image_quality": {
                    "sufficient_for_analysis": True,
                    "blur": "none",
                    "glare": "none",
                    "subject_fills_frame": True,
                    "text_readability": "clear",
                },
                "items": [],
                "unresolved_text": [],
                "uncertainty_reasons": [],
                "next_actions": [],
            }
        )


def test_unknown_medication_image_cannot_contain_items() -> None:
    with pytest.raises(ValidationError, match="cannot contain medication items"):
        MedicationImageAnalysis.model_validate(
            {
                "subject_type": "unknown",
                "document_type": "none",
                "image_quality": {
                    "sufficient_for_analysis": True,
                    "blur": "none",
                    "glare": "none",
                    "subject_fills_frame": True,
                    "text_readability": "none",
                },
                "items": [
                    {
                        "product_name": "",
                        "generic_name": "",
                        "strength": "",
                        "dosage_form": "",
                        "permit_number": "",
                        "nhi_code": "",
                        "manufacturer": "",
                        "directions": {
                            "dose": "",
                            "frequency": "",
                            "route": "",
                            "duration": "",
                            "quantity": "",
                            "instructions": [],
                        },
                        "source_text": [],
                        "confidence": "low",
                        "evidence": {
                            "dosage_form": "unknown",
                            "colors": [],
                            "shape": "unknown",
                            "score_marks": [],
                            "symbols_or_logos": [],
                            "imprints": [],
                            "package_text": [],
                            "distinctive_features": [],
                        },
                    }
                ],
                "unresolved_text": [],
                "uncertainty_reasons": [],
                "next_actions": [],
            }
        )
