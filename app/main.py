"""Candidate v2 API. Staging only until the blinded external audit is complete."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import resource
import secrets
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

import requests
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer

from app.logic import (
    cns_union_decision,
    keyword_evidence,
    keyword_only_decision,
    normalize_cancer_type,
    review_trigger_reasons,
)


LOGGER = logging.getLogger("cognitive_classifier")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

API_VERSION = "2.0.0-rc2"
MODEL_REPO = os.environ.get("MODEL_REPO", "Mubashir-ZZ/cognitive-classifier-v7-cns")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "cns_v7_quantized.pt")
MODEL_REVISION = os.environ.get("MODEL_REVISION")
MODEL_DIR = os.environ.get("MODEL_DIR")
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/app/hybrid_config.json"))
HF_TOKEN = os.environ.get("HF_TOKEN")
BUILD_COMMIT = os.environ.get("RENDER_GIT_COMMIT", os.environ.get("BUILD_COMMIT", "unknown"))
REVIEW_QUEUE_WEBHOOK = os.environ.get("REVIEW_QUEUE_WEBHOOK")
CLASSIFIER_API_KEY = os.environ.get("CLASSIFIER_API_KEY")
MAX_BATCH_SIZE = 6
MAX_TEXT_CHARACTERS = 100_000
MAX_REQUEST_BYTES = 750_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenizer_sha256(value) -> str:
    backend = getattr(value, "backend_tokenizer", None)
    if backend is not None:
        return sha256_text(backend.to_str())
    payload = json.dumps(
        {"vocabulary": value.get_vocab(), "special_tokens": value.special_tokens_map},
        sort_keys=True,
        ensure_ascii=False,
    )
    return sha256_text(payload)


def peak_rss_mb() -> float:
    """Return process peak resident memory in MiB on Linux and macOS."""
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_used = peak if sys.platform == "darwin" else peak * 1024
    return round(bytes_used / (1024 * 1024), 3)


with CONFIG_PATH.open(encoding="utf-8") as handle:
    CONFIG = json.load(handle)
KEYWORDS: list[str] = CONFIG["keywords"]
REVIEW_TRIGGER_PATTERNS: list[str] = CONFIG["review_trigger_patterns"]
BERT_THRESHOLD = float(CONFIG["bert_threshold"])
BERT_MAX_TOKENS = int(CONFIG["bert_max_tokens"])
UNCERTAIN_LOW, UNCERTAIN_HIGH = map(float, CONFIG["uncertainty_zone"])
DECISION_RULE_VERSION = str(CONFIG["decision_rule_version"])
CONFIG_SHA256 = sha256_file(CONFIG_PATH)

if MODEL_DIR:
    model_dir = Path(MODEL_DIR)
    tokenizer_source = str(model_dir)
    model_path = model_dir / MODEL_FILENAME
else:
    tokenizer_source = MODEL_REPO
    model_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            token=HF_TOKEN,
            revision=MODEL_REVISION,
        )
    )

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=HF_TOKEN, revision=MODEL_REVISION)
device = torch.device("cpu")
traced_model = torch.jit.load(str(model_path), map_location=device)
traced_model.eval()
MODEL_SHA256 = sha256_file(model_path)
TOKENIZER_SHA256 = tokenizer_sha256(tokenizer)
REVIEW_DELIVERY_FAILURES = 0


class KeywordMatch(BaseModel):
    term: str
    start: int
    end: int


class PredictionRequest(BaseModel):
    cancer_type: Literal["CNS", "Breast", "Lung", "HeadNeck"]
    outcome_text: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)
    trial_id: str | None = Field(default=None, max_length=128)

    @field_validator("cancer_type", mode="before")
    @classmethod
    def canonicalize_cancer_type(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("cancer_type must be text")
        return normalize_cancer_type(value)

    @field_validator("outcome_text")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("outcome_text cannot be empty")
        return value


class PredictionResponse(BaseModel):
    request_id: str
    trial_id: str | None
    cancer_type: Literal["CNS", "Breast", "Lung", "HeadNeck"]
    predicted_cognitive: bool
    decision_basis: Literal["bert_and_keyword", "bert_only", "keyword_only", "neither"]
    bert_probability: float | None
    bert_positive: bool | None
    keyword_hit: bool
    keyword_matches: list[KeywordMatch]
    review_recommended: bool
    auto_label_eligible: bool
    review_reasons: list[str]
    source_text_sha256: str
    text_characters: int
    bert_input_tokens: int | None
    bert_truncated: bool | None
    api_version: str
    decision_rule_version: str
    model_sha256: str
    tokenizer_sha256: str
    keyword_config_sha256: str
    build_commit: str


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


app = FastAPI(
    title="Neurocognitive Outcome Classifier",
    description="Staging candidate for registry-outcome research screening; not clinical decision support.",
    version=API_VERSION,
)


@app.exception_handler(RequestValidationError)
async def non_echoing_validation_error(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = [
        {"location": list(error.get("loc", ())), "type": error.get("type", "validation_error")}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": "Request validation failed", "errors": errors})


@app.middleware("http")
async def request_security_controls(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request body is too large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
    if CLASSIFIER_API_KEY and request.url.path not in {"/health", "/about", "/openapi.json", "/docs"}:
        provided = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(provided, CLASSIFIER_API_KEY):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


def extract_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, tuple):
        return output[0]
    return output.logits


def bert_probabilities(texts: list[str]) -> list[tuple[float, int, bool]]:
    full_token_lengths = [len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in texts]
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=BERT_MAX_TOKENS,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        output = traced_model(inputs["input_ids"], inputs["attention_mask"])
    logits = extract_logits(output)
    probabilities = torch.softmax(logits, dim=1)[:, 1].tolist()
    return [
        (probability, min(token_count, BERT_MAX_TOKENS), token_count > BERT_MAX_TOKENS)
        for probability, token_count in zip(probabilities, full_token_lengths)
    ]


def version_fields() -> dict:
    return {
        "api_version": API_VERSION,
        "decision_rule_version": DECISION_RULE_VERSION,
        "model_sha256": MODEL_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "keyword_config_sha256": CONFIG_SHA256,
        "build_commit": BUILD_COMMIT,
    }


def make_response(req: PredictionRequest, bert_result: tuple[float, int, bool] | None) -> PredictionResponse:
    text = req.outcome_text
    evidence = keyword_evidence(text, KEYWORDS)
    trigger_reasons = review_trigger_reasons(text, REVIEW_TRIGGER_PATTERNS)
    if req.cancer_type == "CNS":
        if bert_result is None:
            raise RuntimeError("CNS prediction requires a BERT probability")
        bert_probability, bert_input_tokens, bert_truncated = bert_result
        if bert_truncated:
            trigger_reasons.append("BERT input exceeded the configured token window")
        decision = cns_union_decision(
            bert_probability,
            bool(evidence),
            trigger_reasons,
            threshold=BERT_THRESHOLD,
            uncertain_low=UNCERTAIN_LOW,
            uncertain_high=UNCERTAIN_HIGH,
        )
        bert_positive: bool | None = bert_probability >= BERT_THRESHOLD
    else:
        decision = keyword_only_decision(bool(evidence), trigger_reasons)
        bert_positive = None
        bert_probability = None
        bert_input_tokens = None
        bert_truncated = None

    response = PredictionResponse(
        request_id=str(uuid4()),
        trial_id=req.trial_id,
        cancer_type=req.cancer_type,
        predicted_cognitive=decision.predicted_cognitive,
        decision_basis=decision.decision_basis,
        bert_probability=None if bert_probability is None else round(bert_probability, 6),
        bert_positive=bert_positive,
        keyword_hit=bool(evidence),
        keyword_matches=[KeywordMatch(term=item.term, start=item.start, end=item.end) for item in evidence],
        review_recommended=decision.review_recommended,
        auto_label_eligible=not decision.review_recommended,
        review_reasons=list(decision.review_reasons),
        source_text_sha256=sha256_text(text),
        text_characters=len(text),
        bert_input_tokens=bert_input_tokens,
        bert_truncated=bert_truncated,
        **version_fields(),
    )
    if response.review_recommended:
        log_review_event(req, response)
    return response


def log_review_event(req: PredictionRequest, response: PredictionResponse) -> None:
    global REVIEW_DELIVERY_FAILURES
    if not REVIEW_QUEUE_WEBHOOK:
        return
    payload = {
        "trial_id": req.trial_id,
        "cancer_type": req.cancer_type,
        "source_text_sha256": response.source_text_sha256,
        "predicted_cognitive": response.predicted_cognitive,
        "decision_basis": response.decision_basis,
        "bert_probability": response.bert_probability,
        "keyword_matches": [item.model_dump() for item in response.keyword_matches],
        "review_reasons": response.review_reasons,
        **version_fields(),
    }
    try:
        result = requests.post(REVIEW_QUEUE_WEBHOOK, json=payload, timeout=5)
        result.raise_for_status()
    except requests.RequestException as exc:
        REVIEW_DELIVERY_FAILURES += 1
        LOGGER.warning("review queue delivery failed: %s", type(exc).__name__)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    bert_result = bert_probabilities([req.outcome_text])[0] if req.cancer_type == "CNS" else None
    return make_response(req, bert_result)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictionRequest) -> BatchPredictionResponse:
    cns_indices = [index for index, item in enumerate(req.items) if item.cancer_type == "CNS"]
    result_by_index: dict[int, tuple[float, int, bool]] = {}
    if cns_indices:
        bert_results = bert_probabilities([req.items[index].outcome_text for index in cns_indices])
        result_by_index.update(zip(cns_indices, bert_results))
    results = [make_response(item, result_by_index.get(index)) for index, item in enumerate(req.items)]
    return BatchPredictionResponse(results=results)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": str(device),
        "prediction_authentication_enabled": bool(CLASSIFIER_API_KEY),
        "review_queue_configured": bool(REVIEW_QUEUE_WEBHOOK),
        "review_delivery_failures_total": REVIEW_DELIVERY_FAILURES,
        **version_fields(),
    }


@app.get("/about")
def about() -> dict:
    return {
        "purpose": "Research screening of clinical-trial registry outcome text; not clinical decision support.",
        "deployment_status": "candidate staging build; production promotion is blocked on frozen independent human validation",
        "cns_candidate_rule": "BERT v7 positive OR 66-term keyword detector positive; disagreements require review",
        "non_cns_rule": "66-term case-insensitive substring detector",
        "validation_status": (
            "Previous 120-case figures were development checks contaminated by v7 training overlap and are not "
            "external-validation estimates. A disjoint, design-weighted 300-trial challenge set is awaiting independent human labels."
        ),
        "bert_threshold": BERT_THRESHOLD,
        "bert_max_tokens": BERT_MAX_TOKENS,
        "bert_input_policy": "Head-only truncation at the configured token limit; full text is still scanned by the keyword detector and truncation is disclosed per response.",
        "maximum_text_characters": MAX_TEXT_CHARACTERS,
        "maximum_request_bytes": MAX_REQUEST_BYTES,
        **version_fields(),
    }


@app.get("/diagnostics")
def diagnostics() -> dict:
    """Authenticated staging measurements without request or outcome text."""
    return {
        "peak_rss_mb": peak_rss_mb(),
        "review_delivery_failures_total": REVIEW_DELIVERY_FAILURES,
        "prediction_authentication_enabled": bool(CLASSIFIER_API_KEY),
        **version_fields(),
    }
