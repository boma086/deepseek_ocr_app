import os
import re
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import httpx
import uvicorn
from decouple import config as env_config


REMOTE_OCR_API_URL = env_config("REMOTE_OCR_API_URL", default="").strip()
REMOTE_OCR_API_KEY = env_config("REMOTE_OCR_API_KEY", default="").strip()
REMOTE_OCR_MODEL = env_config("REMOTE_OCR_MODEL", default="deepseek-ai/DeepSeek-OCR").strip()
_REMOTE_OCR_TIMEOUT_RAW = env_config("REMOTE_OCR_TIMEOUT", default="60")
try:
    REMOTE_OCR_TIMEOUT = int(str(_REMOTE_OCR_TIMEOUT_RAW).strip())
except (TypeError, ValueError):
    REMOTE_OCR_TIMEOUT = 60


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="DeepSeek-OCR Cloud API",
    description="Cloud-based OCR via SiliconFlow API",
    version="1.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Prompt builder
# -----------------------------
def build_prompt(
    mode: str,
    user_prompt: str,
    grounding: bool,
    find_term: Optional[str],
    schema: Optional[str],
    include_caption: bool,
) -> str:
    """Build the prompt based on mode"""
    parts: List[str] = ["<image>"]
    mode_requires_grounding = mode in {"find_ref", "layout_map", "pii_redact"}
    if grounding or mode_requires_grounding:
        parts.append("<|grounding|>")

    instruction = ""
    if mode == "plain_ocr":
        instruction = "Free OCR. Only output the raw text."
    elif mode == "markdown":
        instruction = "Convert the document to markdown."
    elif mode == "tables_csv":
        instruction = (
            "Extract every table and output CSV only. "
            "Use commas, minimal quoting. If multiple tables, separate with a line containing '---'."
        )
    elif mode == "tables_md":
        instruction = "Extract every table as GitHub-flavored Markdown tables. Output only the tables."
    elif mode == "kv_json":
        schema_text = schema.strip() if schema else "{}"
        instruction = (
            "Extract key fields and return strict JSON only. "
            f"Use this schema (fill the values): {schema_text}"
        )
    elif mode == "figure_chart":
        instruction = (
            "Parse the figure. First extract any numeric series as a two-column table (x,y). "
            "Then summarize the chart in 2 sentences. Output the table, then a line '---', then the summary."
        )
    elif mode == "find_ref":
        key = (find_term or "").strip() or "Total"
        instruction = f"Locate <|ref|>{key}<|/ref|> in the image."
    elif mode == "layout_map":
        instruction = (
            'Return a JSON array of blocks with fields {"type":["title","paragraph","table","figure"],'
            '"box":[x1,y1,x2,y2]}. Do not include any text content.'
        )
    elif mode == "pii_redact":
        instruction = (
            'Find all occurrences of emails, phone numbers, postal addresses, and IBANs. '
            'Return a JSON array of objects {label, text, box:[x1,y1,x2,y2]}.'
        )
    elif mode == "multilingual":
        instruction = "Free OCR. Detect the language automatically and output in the same script."
    elif mode == "describe":
        instruction = "Describe this image. Focus on visible key elements."
    elif mode == "freeform":
        instruction = user_prompt.strip() if user_prompt else "OCR this image."
    else:
        instruction = "OCR this image."

    if include_caption and mode not in {"describe"}:
        instruction = instruction + "\nThen add a one-paragraph description of the image."

    parts.append(instruction)
    return "\n".join(parts)


# -----------------------------
# Grounding parser utilities
# -----------------------------
DET_BLOCK = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>\s*(?P<coords>\[.*\])\s*<\|/det\|>",
    re.DOTALL,
)


def clean_grounding_text(text: str) -> str:
    cleaned = re.sub(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*\[.*\]\s*<\|/det\|>",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r"<\|grounding\|>", "", cleaned)
    return cleaned.strip()


def parse_detections(text: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    for m in DET_BLOCK.finditer(text or ""):
        label = m.group("label").strip()
        coords_str = m.group("coords").strip()

        try:
            import ast

            parsed = ast.literal_eval(coords_str)

            if (
                isinstance(parsed, list)
                and len(parsed) == 4
                and all(isinstance(n, (int, float)) for n in parsed)
            ):
                box_coords = [parsed]
            elif isinstance(parsed, list):
                box_coords = parsed
            else:
                raise ValueError("Unsupported coords structure")

            for box in box_coords:
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    x1 = int(float(box[0]) / 999 * image_width)
                    y1 = int(float(box[1]) / 999 * image_height)
                    x2 = int(float(box[2]) / 999 * image_width)
                    y2 = int(float(box[3]) / 999 * image_height)
                    boxes.append({"label": label, "box": [x1, y1, x2, y2]})
        except Exception:
            continue

    return boxes


def _normalize_remote_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return _normalize_remote_content(content["text"])
        if "content" in content:
            return _normalize_remote_content(content["content"])
        parts = []
        for value in content.values():
            normalized = _normalize_remote_content(value)
            if normalized:
                parts.append(normalized)
        return "\n".join(parts).strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            normalized = _normalize_remote_content(item)
            if normalized:
                parts.append(normalized)
        return "\n".join(parts).strip()
    return str(content).strip()


async def invoke_remote_ocr(prompt_text: str, image_bytes: bytes, content_type: Optional[str]) -> str:
    if not REMOTE_OCR_API_URL or not REMOTE_OCR_API_KEY:
        raise HTTPException(status_code=500, detail="Remote OCR API is not configured")

    media_type = (content_type or "image/png").split(";")[0] or "image/png"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{media_type};base64,{image_b64}"

    payload: Dict[str, Any] = {
        "model": REMOTE_OCR_MODEL or "deepseek-ai/DeepSeek-OCR",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {REMOTE_OCR_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=REMOTE_OCR_TIMEOUT) as client:
            response = await client.post(REMOTE_OCR_API_URL, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Remote OCR request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text[:500]
        raise HTTPException(status_code=response.status_code, detail=f"Remote OCR error: {detail}")

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Remote OCR returned invalid JSON") from exc

    text = ""
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices:
        choice0 = choices[0] or {}
        message = choice0.get("message", {}) if isinstance(choice0, dict) else {}
        text = _normalize_remote_content(message.get("content"))
        if not text and "delta" in choice0:
            text = _normalize_remote_content(choice0.get("delta"))

    if not text and isinstance(data, dict):
        fallback_candidates = [
            data.get("content"),
            data.get("result"),
            data.get("text"),
            data.get("output"),
        ]
        for candidate in fallback_candidates:
            text = _normalize_remote_content(candidate)
            if text:
                break

    if not text:
        raise HTTPException(status_code=502, detail="Remote OCR returned empty response")

    return text


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    return {"message": "DeepSeek-OCR Cloud API is running! ☁️", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy", "provider": "cloud"}


@app.post("/api/ocr")
async def ocr_inference(
    image: UploadFile = File(...),
    mode: str = Form("plain_ocr"),
    prompt: str = Form(""),
    grounding: bool = Form(False),
    include_caption: bool = Form(False),
    find_term: Optional[str] = Form(None),
    schema: Optional[str] = Form(None),
    base_size: int = Form(1024),
    image_size: int = Form(640),
    crop_mode: bool = Form(True),
    test_compress: bool = Form(False),
):
    if not REMOTE_OCR_API_URL or not REMOTE_OCR_API_KEY:
        raise HTTPException(status_code=500, detail="Remote OCR API is not configured")

    prompt_text = build_prompt(
        mode=mode,
        user_prompt=prompt,
        grounding=grounding,
        find_term=find_term,
        schema=schema,
        include_caption=include_caption,
    )

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        with Image.open(BytesIO(content)) as im:
            orig_w, orig_h = im.size
    except Exception:
        orig_w = orig_h = None

    remote_text = await invoke_remote_ocr(
        prompt_text=prompt_text,
        image_bytes=content,
        content_type=image.content_type,
    )
    raw_text = remote_text.strip() or "No text returned by remote OCR."

    boxes = parse_detections(raw_text, orig_w or 1, orig_h or 1) if ("<|det|>" in raw_text or "<|ref|>" in raw_text) else []
    display_text = clean_grounding_text(raw_text) if ("<|ref|>" in raw_text or "<|grounding|>" in raw_text) else raw_text

    if not display_text and boxes:
        display_text = ", ".join([b["label"] for b in boxes])

    return JSONResponse({
        "success": True,
        "text": display_text,
        "raw_text": raw_text,
        "boxes": boxes,
        "image_dims": {"w": orig_w, "h": orig_h},
        "metadata": {
            "mode": mode,
            "grounding": grounding or (mode in {"find_ref", "layout_map", "pii_redact"}),
            "base_size": base_size,
            "image_size": image_size,
            "crop_mode": crop_mode,
            "provider": "cloud",
        },
    })


if __name__ == "__main__":
    host = env_config("API_HOST", default="0.0.0.0")
    port = env_config("API_PORT", default=8000, cast=int)
    uvicorn.run(app, host=host, port=port)
