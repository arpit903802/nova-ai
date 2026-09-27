import os
import time
import requests

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

# =========================================================
# NVIDIA API
# =========================================================

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
NVIDIA_IMAGE_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell"


# =========================================================
# API KEYS
# =========================================================

API_KEYS = [
    os.getenv("NVIDIA_API_KEY_1"),
    os.getenv("NVIDIA_API_KEY_2"),
    os.getenv("NVIDIA_API_KEY_3"),
    os.getenv("NVIDIA_API_KEY_4"),
    os.getenv("NVIDIA_API_KEY_5"),
    os.getenv("NVIDIA_API_KEY_6"),
]

# Empty values remove karo
API_KEYS = [key.strip() for key in API_KEYS if key and key.strip()]


# =========================================================
# MODELS
# =========================================================

# Tumhare testing results ke basis par primary/fallback models
TEXT_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "mistralai/mistral-nemotron",
    "meta/muse-glimmer-30b",
]

VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"

IMAGE_MODEL = "black-forest-labs/flux.1-schnell"


# =========================================================
# INTERNAL STATE
# =========================================================

key_index = 0


# =========================================================
# HELPERS
# =========================================================

def get_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def get_next_key():
    global key_index

    if not API_KEYS:
        return None

    key = API_KEYS[key_index % len(API_KEYS)]
    key_index += 1

    return key


def clean_messages(messages):
    """
    Frontend se aane wale messages ko basic validation ke saath
    clean karta hai.
    """

    if not isinstance(messages, list):
        return []

    cleaned = []

    for message in messages[-20:]:

        if not isinstance(message, dict):
            continue

        role = message.get("role")

        if role not in ["system", "user", "assistant"]:
            continue

        content = message.get("content")

        if content is None:
            continue

        cleaned.append({
            "role": role,
            "content": content
        })

    return cleaned


# =========================================================
# CHAT
# =========================================================

def ask_nvidia(messages):

    if not API_KEYS:
        return {
            "ok": False,
            "error": "No NVIDIA API keys configured on Render."
        }

    last_errors = []

    # Model -> Key fallback
    for model in TEXT_MODELS:

        for attempt in range(len(API_KEYS)):

            api_key = get_next_key()

            if not api_key:
                break

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.5,
                "top_p": 0.9,
                "max_tokens": 1200,
                "stream": False,
            }

            try:

                response = requests.post(
                    NVIDIA_CHAT_URL,
                    headers=get_headers(api_key),
                    json=payload,
                    timeout=25,
                )

                # Successful response
                if response.status_code == 200:

                    data = response.json()

                    choices = data.get("choices", [])

                    if choices:

                        message = choices[0].get("message", {})

                        answer = message.get("content")

                        if answer:

                            return {
                                "ok": True,
                                "answer": answer,
                                "model": model,
                            }

                    last_errors.append(
                        f"{model}: empty response"
                    )

                    continue

                # Rate limit / temporary server errors
                if response.status_code in [408, 429, 500, 502, 503, 504]:

                    last_errors.append(
                        f"{model}: HTTP {response.status_code}"
                    )

                    continue

                # Invalid model/key/etc.
                last_errors.append(
                    f"{model}: HTTP {response.status_code}"
                )

            except requests.exceptions.Timeout:

                last_errors.append(
                    f"{model}: timeout"
                )

            except requests.exceptions.RequestException as error:

                last_errors.append(
                    f"{model}: {str(error)[:120]}"
                )

            except Exception as error:

                last_errors.append(
                    f"{model}: {str(error)[:120]}"
                )

    return {
        "ok": False,
        "error": "All NVIDIA API attempts failed.",
        "details": last_errors[-15:]
    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return send_from_directory(
        app.static_folder,
        "index.html"
    )


# =========================================================
# HEALTH
# =========================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "online",
        "keys_configured": len(API_KEYS),
        "text_models": TEXT_MODELS,
        "vision_model": VISION_MODEL,
        "image_model": IMAGE_MODEL,
    })


# =========================================================
# NVIDIA MODELS CHECK
# =========================================================

@app.route("/api/models", methods=["GET"])
def models():

    if not API_KEYS:

        return jsonify({
            "ok": False,
            "error": "No API keys configured."
        }), 500

    api_key = get_next_key()

    try:

        response = requests.get(
            NVIDIA_MODELS_URL,
            headers=get_headers(api_key),
            timeout=15,
        )

        return (
            response.text,
            response.status_code,
            {
                "Content-Type": "application/json"
            }
        )

    except Exception as error:

        return jsonify({
            "ok": False,
            "error": str(error)
        }), 500


# =========================================================
# CHAT API
# =========================================================

@app.route("/api/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "ok": False,
                "error": "Invalid JSON request."
            }), 400

        messages = clean_messages(
            data.get("messages", [])
        )

        if not messages:

            return jsonify({
                "ok": False,
                "error": "No valid messages provided."
            }), 400

        result = ask_nvidia(messages)

        if not result["ok"]:

            return jsonify(result), 502

        return jsonify(result)

    except Exception as error:

        return jsonify({
            "ok": False,
            "error": str(error)
        }), 500


# =========================================================
# VISION / IMAGE UNDERSTANDING
# =========================================================

@app.route("/api/vision", methods=["POST"])
def vision():

    try:

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "ok": False,
                "error": "Invalid JSON request."
            }), 400

        image = data.get("image")
        prompt = data.get(
            "prompt",
            "Describe this image accurately."
        )

        if not image:

            return jsonify({
                "ok": False,
                "error": "Image is required."
            }), 400

        if not API_KEYS:

            return jsonify({
                "ok": False,
                "error": "No NVIDIA API keys configured."
            }), 500

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        }
                    }
                ]
            }
        ]

        last_error = ""

        for _ in range(len(API_KEYS)):

            api_key = get_next_key()

            payload = {
                "model": VISION_MODEL,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1000,
                "stream": False,
            }

            try:

                response = requests.post(
                    NVIDIA_CHAT_URL,
                    headers=get_headers(api_key),
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:

                    result = response.json()

                    choices = result.get(
                        "choices",
                        []
                    )

                    if choices:

                        answer = choices[0].get(
                            "message",
                            {}
                        ).get("content", "")

                        if answer:

                            return jsonify({
                                "ok": True,
                                "answer": answer,
                                "model": VISION_MODEL
                            })

                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )

            except requests.exceptions.Timeout:

                last_error = "Vision request timeout."

            except Exception as error:

                last_error = str(error)

        return jsonify({
            "ok": False,
            "error": last_error
        }), 502

    except Exception as error:

        return jsonify({
            "ok": False,
            "error": str(error)
        }), 500


# =========================================================
# IMAGE GENERATION
# =========================================================

@app.route("/api/image", methods=["POST"])
def image_generation():

    try:

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "ok": False,
                "error": "Invalid JSON request."
            }), 400

        prompt = str(
            data.get("prompt", "")
        ).strip()

        if not prompt:

            return jsonify({
                "ok": False,
                "error": "Prompt is required."
            }), 400

        if not API_KEYS:

            return jsonify({
                "ok": False,
                "error": "No NVIDIA API keys configured."
            }), 500

        last_error = ""

        for _ in range(len(API_KEYS)):

            api_key = get_next_key()

            payload = {
                "prompt": prompt,
                "steps": 4,
                "width": 1024,
                "height": 1024,
                "seed": int(time.time()) % 2147483647,
            }

            try:

                response = requests.post(
                    NVIDIA_IMAGE_URL,
                    headers=get_headers(api_key),
                    json=payload,
                    timeout=90,
                )

                if response.status_code == 200:

                    result = response.json()

                    # NVIDIA GenAI artifact format
                    artifacts = result.get(
                        "artifacts",
                        []
                    )

                    if artifacts:

                        image_b64 = artifacts[0].get(
                            "base64"
                        )

                        if image_b64:

                            return jsonify({
                                "ok": True,
                                "image":
                                    "data:image/png;base64,"
                                    + image_b64,
                                "model": IMAGE_MODEL
                            })

                    # Alternate response
                    image_b64 = result.get("image")

                    if image_b64:

                        return jsonify({
                            "ok": True,
                            "image":
                                "data:image/png;base64,"
                                + image_b64,
                            "model": IMAGE_MODEL
                        })

                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )

            except requests.exceptions.Timeout:

                last_error = "Image generation timeout."

            except requests.exceptions.RequestException as error:

                last_error = str(error)

            except Exception as error:

                last_error = str(error)

        return jsonify({
            "ok": False,
            "error": last_error
        }), 502

    except Exception as error:

        return jsonify({
            "ok": False,
            "error": str(error)
        }), 500


# =========================================================
# 404
# =========================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "ok": False,
        "error": "Endpoint not found."
    }), 404


# =========================================================
# 500
# =========================================================

@app.errorhandler(500)
def server_error(error):

    return jsonify({
        "ok": False,
        "error": "Internal server error."
    }), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.getenv("PORT", "10000")
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
                )
