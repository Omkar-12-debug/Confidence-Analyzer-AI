from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(backend_path)

import config
from integration.orchestrator import run_pipeline, run_pipeline_from_file

app = Flask(__name__)
CORS(app)  # Allow frontend dev server to call API

ALLOWED_EXTENSIONS = {'.webm', '.mp4', '.avi', '.mov', '.mkv', '.wav', '.ogg'}


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "Confidence Analyzer AI Backend",
        "status": "running",
        "endpoints": ["/api/health", "/api/analyze", "/api/latest-result", "/api/history"]
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Confidence Analyzer API is running."})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Accepts either:
    1. A multipart/form-data upload with a 'video' file field (real recording)
    2. A JSON payload with audio_features and facial_features (legacy/testing)
    """
    # Check if this is a file upload
    if 'video' in request.files:
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        logger.info(f"Upload received: filename={file.filename}, content_type={file.content_type}")

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"upload_{timestamp}{ext}"
        upload_path = config.UPLOADS_DIR / safe_name
        file.save(str(upload_path))

        # Validate file size
        file_size = os.path.getsize(str(upload_path))
        logger.info(f"Saved upload: {upload_path} ({file_size} bytes, ext={ext})")

        if file_size == 0:
            os.remove(str(upload_path))
            return jsonify({"error": "Uploaded file is empty"}), 400

        source_id = request.form.get("source_id", "recording")

        try:
            result = run_pipeline_from_file(str(upload_path), source_id=source_id)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up uploaded file after processing
            try:
                if os.path.exists(str(upload_path)):
                    os.remove(str(upload_path))
            except:
                pass

    # Legacy JSON mode
    elif request.is_json:
        payload = request.json
        if not payload:
            return jsonify({"error": "No JSON payload provided."}), 400
        try:
            result = run_pipeline(payload)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Send either a 'video' file or JSON payload."}), 400


@app.route('/api/latest-result', methods=['GET'])
def get_latest_result():
    latest_path = config.REPORTS_DIR / "latest_result.json"
    if latest_path.exists():
        with open(latest_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "No latest result found."}), 404


@app.route('/api/history', methods=['GET'])
def get_history():
    from integration.orchestrator import score_to_class

    history = []
    history_dir = config.HISTORY_DIR
    if history_dir.exists():
        files = sorted(history_dir.glob("result_*.json"), reverse=True)
        for f in files[:50]:  # Limit to 50 entries
            try:
                with open(f, "r") as fh:
                    data = json.load(fh)
                    # Build summary entry
                    fusion = data.get("fusion_module", {})
                    audio = data.get("audio_module", {})
                    facial = data.get("facial_module", {})
                    analysis_mode = data.get("analysis_mode",
                                             fusion.get("analysis_mode", "unknown"))
                    score = fusion.get("final_confidence_score")
                    # facial score may be None when no face was detected
                    raw_facial_score = facial.get("score")
                    facial_score_val = None if raw_facial_score is None else raw_facial_score

                    # Derive class from score; handle None for incomplete modes
                    if score is not None:
                        conf_class = score_to_class(score)
                    else:
                        conf_class = "Incomplete"

                    history.append({
                        "id": f.stem,
                        "timestamp": data.get("timestamp", ""),
                        "analysis_mode": analysis_mode,
                        "confidence_class": conf_class,
                        "confidence_score": score,
                        "audio_score": audio.get("score"),
                        "facial_score": facial_score_val,
                        "audio_status": audio.get("status", "unknown"),
                        "facial_status": facial.get("status", "unknown"),
                    })
            except Exception:
                continue
    return jsonify({"history": history})


@app.route('/api/train-fusion', methods=['POST'])
def train_fusion():
    try:
        from fusion_model.train_fusion_model import train_model
        train_model()
        return jsonify({"status": "success", "message": "Fusion model trained successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
