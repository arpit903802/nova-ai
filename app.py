import os
import requests
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODEL = "meta/llama-3.3-70b-instruct"


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        if not NVIDIA_API_KEY:
            return jsonify({
                "error": "NVIDIA_API_KEY is not configured on Render."
            }), 500

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Invalid request."
            }), 400

        messages = data.get("messages", [])

        if not messages:
            return jsonify({
                "error": "No message provided."
            }), 400

        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        response = requests.post(
            NVIDIA_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        try:
            result = response.json()
        except Exception:
            return jsonify({
                "error": "NVIDIA returned an invalid response.",
                "status": response.status_code
            }), 502

        if response.status_code != 200:
            return jsonify({
                "error": result.get("detail")
                or result.get("error")
                or "NVIDIA API request failed.",
                "status": response.status_code
            }), response.status_code

        answer = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        return jsonify({
            "answer": answer
        })

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "NVIDIA API request timed out."
        }), 504

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
