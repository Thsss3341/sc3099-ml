"""
SAIV Face Recognition & Risk Service - Module 3

This is the skeleton implementation for the Face Recognition module.
Students must implement face enrollment, verification, liveness detection,
and risk scoring.

Privacy Requirements:
- NO raw face images should be stored
- Process images in-memory only
- Store only SHA-256 hashes of face embeddings

Recommended Libraries:
- MediaPipe: Face detection and 468-landmark face mesh
- OpenCV: Image processing
- Pillow: Image loading from base64
- NumPy: Numerical operations
"""

import base64
import io
import ipaddress
import os
import logging

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import mediapipe as mp
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.70"))
FACE_MISMATCH_RISK_PENALTY = float(os.getenv("FACE_MISMATCH_RISK_PENALTY", "0.40"))
FACE_HASH_SIMILARITY_MIN = float(os.getenv("FACE_HASH_SIMILARITY_MIN", "0.64"))
FACE_SIM_LOGISTIC_CENTER = float(os.getenv("FACE_SIM_LOGISTIC_CENTER", "0.68"))
FACE_SIM_LOGISTIC_STEEPNESS = float(os.getenv("FACE_SIM_LOGISTIC_STEEPNESS", "26.0"))
MOBILEFACENET_MODEL_PATH = os.getenv("MOBILEFACENET_MODEL_PATH", "app/models/w600k_mbf.onnx")

_mobilefacenet_session = None
_mobilefacenet_input_name = None
_mobilefacenet_output_name = None
_mobilefacenet_input_height = 112
_mobilefacenet_input_width = 112
_mobilefacenet_input_layout = "NCHW"

app = FastAPI(
    title="SAIV Face Recognition Service",
    description="Face enrollment, verification, liveness detection, and risk scoring service",
    version="1.0.0"
)

HTTP_REQUEST_TOTAL = Counter(
    "saiv_ml_http_requests_total",
    "Total HTTP requests handled by the ML service",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "saiv_ml_http_request_duration_seconds",
    "HTTP request latency in seconds for the ML service",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalized_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or request.url.path


@app.middleware("http")
async def capture_request_metrics(request: Request, call_next):
    path = _normalized_path(request)
    timer = HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, path=path).time()
    with timer:
        response = await call_next(request)

    HTTP_REQUEST_TOTAL.labels(
        method=request.method,
        path=path,
        status=str(response.status_code),
    ).inc()
    return response

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class FaceEnrollRequest(BaseModel):
    """Request model for face enrollment."""
    user_id: str
    image: str  # Base64 encoded image
    camera_consent: bool = False


class FaceEnrollResponse(BaseModel):
    """Response model for face enrollment."""
    enrollment_successful: bool
    face_template_hash: str  # 64-char SHA-256 hex string
    quality_score: float  # 0.0 to 1.0
    details: Dict[str, Any]


class FaceVerifyRequest(BaseModel):
    """Request model for face verification."""
    image: str  # Base64 encoded image
    reference_template_hash: str  # Hash from enrollment


class FaceVerifyResponse(BaseModel):
    """Response model for face verification."""
    match_passed: bool
    match_score: float  # 0.0 to 1.0
    match_threshold: float  # Default: 0.70
    face_detected: bool
    current_template_hash: str


class LivenessRequest(BaseModel):
    """Request model for liveness check."""
    challenge_response: str  # Base64 encoded image
    challenge_type: str = "passive"  # head_turn, passive


class LivenessResponse(BaseModel):
    """Response model for liveness check."""
    liveness_passed: bool
    liveness_score: float  # 0.0 to 1.0
    liveness_threshold: float  # Default: 0.60
    face_embedding_hash: str
    details: Dict[str, Any]


class GeolocationData(BaseModel):
    """Geolocation data for risk assessment."""
    latitude: float
    longitude: float
    accuracy: float


class RiskAssessRequest(BaseModel):
    """Request model for risk assessment."""
    liveness_score: Optional[float] = None
    face_match_score: Optional[float] = None
    device_signature: Optional[str] = None
    device_public_key: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    geolocation: Optional[GeolocationData] = None


class RiskAssessResponse(BaseModel):
    """Response model for risk assessment."""
    risk_score: float  # 0.0 to 1.0
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    pass_threshold: bool
    risk_threshold: float  # Default: 0.50
    signal_breakdown: Dict[str, float]
    recommendations: List[str]


# =============================================================================
# HEALTH & ROOT ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "SAIV Face Recognition & Risk Service"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint for scraping ML service telemetry."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """List available endpoints."""
    return {
        "service": "SAIV Face Recognition & Risk Service",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/face/enroll",
            "/face/verify",
            "/face/match",
            "/liveness/check",
            "/risk/assess"
        ]
    }


# =============================================================================
# FACE ENROLLMENT ENDPOINT (REQUIRED - 4 points in public tests)
# =============================================================================

@app.post("/face/enroll", response_model=FaceEnrollResponse, status_code=201)
async def enroll_face(request: FaceEnrollRequest):
    """
    Enroll a user's face for future verification.

    TODO: Implement the following:
    1. Validate camera_consent is True (return 400 if False)
    2. Decode base64 image to numpy array
    3. Detect face using MediaPipe FaceDetection
    4. If no face detected, return 400 with "No face detected"
    5. Extract face features/embedding
    6. Generate SHA-256 hash of embedding (64 hex chars)
    7. Calculate quality score based on:
       - Face detection confidence
       - Image resolution
       - Face size relative to image
    8. Return enrollment response

    Success Criteria:
    - Face detected with confidence >= 0.7
    - Quality score >= 0.5
    - Returns 64-char SHA-256 hex hash
    """
    # 1. Validate camera_consent
    if not request.camera_consent:
        raise HTTPException(status_code=400, detail="Camera consent is required")
        
    # 2. Decode base64 image to numpy array
    image_array = decode_base64_image(request.image)
    if image_array is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
        
    # 3. Detect face using MediaPipe FaceDetection
    detection = detect_face(image_array)
    
    # 4. If no face detected, return 400
    if not detection:
        raise HTTPException(status_code=400, detail="No face detected")
        
    confidence = detection.score[0]
    if confidence < 0.7:
        raise HTTPException(status_code=400, detail=f"Face detection confidence too low ({confidence:.2f} < 0.7)")
        
    # 5. Extract face features/embedding
    embedding = extract_face_embedding(image_array, detection)
    if embedding is None:
        raise HTTPException(status_code=400, detail="Could not extract face features")
        
    # 6. Generate SHA-256 hash of embedding
    face_hash = generate_face_hash(embedding)
    
    # 7. Calculate quality score
    h, w = image_array.shape[:2]
    resolution_score = min(1.0, (h * w) / (640 * 480))
    
    bbox = detection.location_data.relative_bounding_box
    face_size_ratio = bbox.width * bbox.height
    face_size_score = min(1.0, face_size_ratio / 0.1)
    quality_score = float(0.5 * confidence + 0.3 * face_size_score + 0.2 * resolution_score)
    
    if quality_score < 0.5:
        raise HTTPException(status_code=400, detail=f"Image quality too low (score: {quality_score:.2f})")
        
    # 8. Return enrollment response
    return FaceEnrollResponse(
        enrollment_successful=True,
        face_template_hash=face_hash,
        quality_score=quality_score,
        details={
            "confidence": float(confidence),
            "resolution": f"{w}x{h}",
            "face_size_ratio": float(face_size_ratio)
        }
    )


# =============================================================================
# FACE VERIFICATION ENDPOINT (REQUIRED - 4 points in public tests)
# =============================================================================

@app.post("/face/verify", response_model=FaceVerifyResponse)
async def verify_face(request: FaceVerifyRequest):
    """
    Verify a face against an enrolled template.

    TODO: Implement the following:
    1. Decode base64 image to numpy array
    2. Detect face using MediaPipe FaceDetection
    3. If no face detected, return with face_detected=False
    4. Extract face features/embedding
    5. Generate SHA-256 hash of current face
    6. Compare hashes or embeddings (choose your approach)
    7. Calculate match_score (0.0 to 1.0)
    8. match_passed = (match_score >= 0.70)

    Note: Hash comparison alone gives binary match. For continuous
    scores, consider perceptual hashing or embedding similarity.
    """
    # 1. Decode base64 image
    image_array = decode_base64_image(request.image)
    if image_array is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
        
    # 2 & 3. Detect face
    detection = detect_face(image_array)
    if not detection:
        return FaceVerifyResponse(
            match_passed=False,
            match_score=0.0,
            match_threshold=FACE_MATCH_THRESHOLD,
            face_detected=False,
            current_template_hash=""
        )
        
    # 4. Extract features
    embedding = extract_face_embedding(image_array, detection)
    if embedding is None:
        return FaceVerifyResponse(
            match_passed=False,
            match_score=0.0,
            match_threshold=FACE_MATCH_THRESHOLD,
            face_detected=True,
            current_template_hash=""
        )
        
    # 5. Generate hash of current face
    current_hash = generate_face_hash(embedding)
    
    # 6 & 7. Compare hashes using Hamming Distance for 256-bit SimHash
    # Convert Hamming similarity to a calibrated score curve for live webcam noise.
    raw_score = 0.0
    similarity = 0.0
    try:
        dist = hamming_distance(current_hash, request.reference_template_hash)
        similarity = max(0.0, min(1.0, 1.0 - (dist / 256.0)))
        # Stricter logistic calibration to reduce false accepts.
        raw_score = float(
            1.0 / (1.0 + np.exp(-FACE_SIM_LOGISTIC_STEEPNESS * (similarity - FACE_SIM_LOGISTIC_CENTER)))
        )
    except (ValueError, TypeError):
        # Fallback: exact hash equality
        if current_hash == request.reference_template_hash:
            raw_score = 1.0
            similarity = 1.0
        else:
            raw_score = 0.0
            similarity = 0.0

    # Apply a mild quality adjustment (instead of hard multiplicative penalty).
    confidence = float(detection.score[0])
    bbox = detection.location_data.relative_bounding_box
    face_size_ratio = max(0.0, float(bbox.width * bbox.height))
    face_size_factor = min(1.0, face_size_ratio / 0.20)
    quality_factor = 0.7 * min(1.0, confidence) + 0.3 * face_size_factor
    quality_multiplier = 0.9 + 0.1 * quality_factor
    match_score = max(0.0, min(1.0, raw_score * quality_multiplier))

    # Hard guardrail: disallow pass when hash similarity is below floor.
    if similarity < FACE_HASH_SIMILARITY_MIN:
        match_score = min(match_score, FACE_MATCH_THRESHOLD - 0.01)

    # 8. Determine success based on threshold
    match_passed = match_score >= FACE_MATCH_THRESHOLD
    
    return FaceVerifyResponse(
        match_passed=match_passed,
        match_score=float(match_score),
        match_threshold=FACE_MATCH_THRESHOLD,
        face_detected=True,
        current_template_hash=current_hash
    )


@app.post("/face/match")
async def match_face(request: FaceVerifyRequest):
    """
    Legacy face matching endpoint. Redirects to /face/verify.
    Kept for backwards compatibility.
    """
    return await verify_face(request)


# =============================================================================
# LIVENESS DETECTION ENDPOINT (REQUIRED - partial; BONUS for advanced)
# =============================================================================

@app.post("/liveness/check", response_model=LivenessResponse)
async def check_liveness(request: LivenessRequest):
    """
    Perform liveness detection on submitted image.

    TODO: Implement the following:
    1. Decode base64 image to numpy array
    2. Detect face using MediaPipe FaceDetection
    3. If no face detected, return with liveness_passed=False
    4. Analyze face for liveness signals:

    REQUIRED (for partial credit):
    - Basic face detection confidence
    - Image quality assessment
    - Face size validation

    BONUS (for extra credit - see API-SPECIFICATION.md):
    - MediaPipe Face Mesh 3D analysis (468 landmarks)
    - Depth cue analysis (nose_tip_z coordinate)
    - Face mesh completeness check
    - Challenge-response detection (blink, head movement)

    Challenge Types:
    - "passive": No user action required (depth/texture analysis)
    - "head_turn": Detect head rotation (face mesh landmarks)

    5. Calculate liveness_score (0.0 to 1.0)
    6. liveness_passed = (liveness_score >= 0.60)
    7. Generate face embedding hash

    Depth Analysis Hints (BONUS):
    - Use MediaPipe FaceMesh to get 3D landmarks
    - nose_tip_z (landmark 1, z-coordinate) indicates depth
    - Real faces: |nose_tip_z| > 0.03 (significant depth)
    - Flat images: |nose_tip_z| < 0.01 (minimal depth)
    """
    # 1. Decode Image
    image_array = decode_base64_image(request.challenge_response)
    if image_array is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 2. Detect face
    detection = detect_face(image_array)
    if not detection:
        return LivenessResponse(
            liveness_passed=False,
            liveness_score=0.0,
            liveness_threshold=0.60,
            face_embedding_hash="",
            details={"error": "No face detected"}
        )

    # 3. Analyze face mesh and depth
    mesh_results = analyze_face_mesh(image_array)
    if not mesh_results["face_mesh_complete"]:
        return LivenessResponse(
            liveness_passed=False,
            liveness_score=0.0,
            liveness_threshold=0.60,
            face_embedding_hash="",
            details={"error": "Face mesh incomplete, unable to assess depth"}
        )

    confidence = detection.score[0]
    bbox = detection.location_data.relative_bounding_box
    face_area = bbox.width * bbox.height
    size_score = min(1.0, face_area / 0.1)

    # Base depth score
    depth_mod = 1.0 if mesh_results["depth_quality"] == "good" else (0.5 if mesh_results["depth_quality"] == "moderate" else 0.0)

    # 4. Challenge checking
    challenge_score = 1.0
    passed_challenge = True
    challenge_details = {}

    if request.challenge_type != "passive":
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
            fm_res = fm.process(image_array)
            if fm_res and fm_res.multi_face_landmarks:
                face_landmarks = fm_res.multi_face_landmarks[0]
                nose = face_landmarks.landmark[1]
                nose_x = nose.x
                nose_y = nose.y

                if request.challenge_type in ("head_left", "head_right", "head_turn"):
                    challenge_details["nose_x"] = nose_x
                    if request.challenge_type == "head_left":
                        # User turns left → nose appears on the RIGHT side of image (x > 0.58)
                        if nose_x <= 0.58:
                            passed_challenge = False
                            challenge_score = 0.5
                    elif request.challenge_type == "head_right":
                        # User turns right → nose appears on the LEFT side of image (x < 0.42)
                        if nose_x >= 0.42:
                            passed_challenge = False
                            challenge_score = 0.5
                    else:  # generic head_turn: either direction
                        if 0.42 <= nose_x <= 0.58:
                            passed_challenge = False
                            challenge_score = 0.5

                elif request.challenge_type == "head_up":
                    challenge_details["nose_y"] = nose_y
                    # Looking up → nose moves to upper portion of frame (smaller y).
                    # Previous cutoff (0.43) was too strict for many webcams/framing setups.
                    # Relax to 0.50 to reduce false negatives while keeping directional intent.
                    if nose_y >= 0.50:
                        passed_challenge = False
                        challenge_score = 0.5

                elif request.challenge_type == "head_down":
                    challenge_details["nose_y"] = nose_y
                    # Looking down → nose moves to lower portion of frame (larger y)
                    if nose_y <= 0.62:
                        passed_challenge = False
                        challenge_score = 0.5
            else:
                passed_challenge = False
                challenge_score = 0.0

    # 5. Final calculations
    base_liveness = (confidence * 0.4) + (size_score * 0.2) + (depth_mod * 0.4)
    liveness_score = float(base_liveness * challenge_score)

    liveness_passed = liveness_score >= 0.60 and passed_challenge

    # 6. Generate embedding hash
    embedding = extract_face_embedding(image_array, detection)
    face_hash = generate_face_hash(embedding) if embedding is not None else ""

    details = {
        "confidence": float(confidence),
        "mesh_depth": float(mesh_results["nose_tip_z"]),
        "depth_quality": mesh_results["depth_quality"],
        "challenge": request.challenge_type,
        "challenge_passed": passed_challenge,
        **challenge_details
    }

    return LivenessResponse(
        liveness_passed=liveness_passed,
        liveness_score=liveness_score,
        liveness_threshold=0.60,
        face_embedding_hash=face_hash,
        details=details
    )


# =============================================================================
# RISK ASSESSMENT ENDPOINT (REQUIRED - 3 points in public tests)
# =============================================================================

@app.post("/risk/assess", response_model=RiskAssessResponse)
async def assess_risk(request: RiskAssessRequest):
    """
    Perform multi-signal risk assessment.

    TODO: Implement the following:
    1. Collect all available signals from request
    2. Calculate individual signal scores (0.0 = safe, 1.0 = risky)
    3. Apply weighted fusion:
       - Liveness: 25%
       - Face match: 25%
       - Device attestation: 20%
       - Network/VPN: 15%
       - Geolocation: 15%
    4. Calculate combined risk_score
    5. Determine risk_level:
       - LOW: risk_score < 0.3
       - MEDIUM: 0.3 <= risk_score < 0.5
       - HIGH: 0.5 <= risk_score < 0.7
       - CRITICAL: risk_score >= 0.7
    6. pass_threshold = (risk_score < 0.50)
    7. Generate recommendations for low-scoring signals

    Signal Analysis:
    - Liveness: Invert score (low liveness = high risk)
    - Face match: Invert score (low match = high risk)
    - Device: Check signature validity, public key format
    - Network: Detect VPN/proxy (private IPs, Tor exit nodes)
    - Geolocation: Check accuracy, validate coordinates

    VPN/Proxy Detection Hints:
    - Private IP ranges: 10.x.x.x, 172.16-31.x.x, 192.168.x.x
    - Check user_agent for VPN indicators
    - High geolocation accuracy (< 10m) might be spoofed
    - Very low accuracy (> 5000m) indicates issues
    """
    signal_breakdown: Dict[str, float] = {}
    recommendations: List[str] = []
    total_risk = 0.0

    # 1. Liveness (25%)
    if request.liveness_score is not None:
        liveness_risk = max(0.0, min(1.0, 1.0 - request.liveness_score)) * 0.25
        signal_breakdown["liveness"] = round(liveness_risk, 3)
        total_risk += liveness_risk
        if request.liveness_score < 0.6:
            recommendations.append("Improve lighting and face visibility")
    else:
        signal_breakdown["liveness"] = 0.0

    # 2. Face match (25%)
    if request.face_match_score is not None:
        face_risk = max(0.0, min(1.0, 1.0 - request.face_match_score)) * 0.25
        signal_breakdown["face_match"] = round(face_risk, 3)
        total_risk += face_risk
        if request.face_match_score < FACE_MATCH_THRESHOLD:
            signal_breakdown["face_policy_penalty"] = round(FACE_MISMATCH_RISK_PENALTY, 3)
            total_risk += FACE_MISMATCH_RISK_PENALTY
            recommendations.append("Re-enroll face or improve image quality")
    else:
        signal_breakdown["face_match"] = 0.0

    # 3. Device attestation (20%)
    has_signature = bool(request.device_signature)
    has_public_key = bool(request.device_public_key)
    if has_signature and has_public_key:
        device_risk = 0.05
    elif has_signature or has_public_key:
        device_risk = 0.10
    else:
        device_risk = 0.20
        recommendations.append("Register or re-bind this device before check-in")
    signal_breakdown["device"] = round(device_risk, 3)
    total_risk += device_risk

    # 4. Network/VPN (15%)
    is_vpn, vpn_confidence = detect_vpn_proxy(request.ip_address, request.user_agent)
    if is_vpn:
        network_risk = min(0.15, 0.15 * vpn_confidence)
        recommendations.append("Disable VPN/proxy and retry check-in")
    else:
        network_risk = 0.0
    signal_breakdown["network"] = round(network_risk, 3)
    total_risk += network_risk

    # 5. Geolocation (15%)
    geo_risk = 0.0
    if request.geolocation:
        acc = request.geolocation.accuracy
        if acc > 5000:
            geo_risk = 0.15
            recommendations.append("Enable precise location services")
        elif acc > 100:
            geo_risk = 0.05
    else:
        geo_risk = 0.10
    signal_breakdown["geolocation"] = round(geo_risk, 3)
    total_risk += geo_risk

    risk_score = round(min(1.0, total_risk), 2)
    if risk_score < 0.3:
        risk_level = "LOW"
    elif risk_score < 0.5:
        risk_level = "MEDIUM"
    elif risk_score < 0.7:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return RiskAssessResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        pass_threshold=risk_score < 0.50,
        risk_threshold=0.50,
        signal_breakdown=signal_breakdown,
        recommendations=recommendations
    )


# =============================================================================
# HELPER FUNCTIONS (Implement these to support your endpoints)
# =============================================================================

def decode_base64_image(base64_string: str):
    """
    Decode a base64 encoded image to a numpy array.

    TODO: Implement using:
    - base64.b64decode()
    - PIL.Image.open(BytesIO(...))
    - numpy.array()

    Handle errors gracefully (invalid base64, corrupt image, etc.)
    """
    try:
        # Remove data URL prefix if present
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB (remove alpha channel if present)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


def detect_face(image_array):
    """
    Detect faces in an image using MediaPipe.

    TODO: Implement using:
    - mediapipe.solutions.face_detection.FaceDetection
    - Return detection results with confidence scores

    Consider setting min_detection_confidence=0.5
    """
    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detection:
        results = face_detection.process(image_array)
        
        if not results.detections:
            print("Detected 0 face(s)")
            return None
            
        num_faces = len(results.detections)
        print(f"Detected {num_faces} face(s)")
        
        if num_faces > 1:
            raise HTTPException(
                status_code=400, 
                detail=f"Multiple faces detected ({num_faces}). Please ensure only one face is in the frame."
            )
        
        # We only work with the first detected face
        detection = results.detections[0]
        
        return detection


def extract_face_embedding(image_array, detection):
    """
    Extract face embedding/features for hashing.

    TODO: Choose your approach:
    - Simple: Crop face region, resize to standard size, flatten
    - Advanced: Use MediaPipe Face Mesh landmarks
    - Even more advanced: Use face recognition model (dlib, etc.)

    Return numpy array that can be hashed.
    """
    if detection is None:
        return None

    bbox = detection.location_data.relative_bounding_box
    h, w = image_array.shape[:2]

    x_min = int(bbox.xmin * w)
    y_min = int(bbox.ymin * h)
    box_w = int(bbox.width * w)
    box_h = int(bbox.height * h)

    # Expand crop a bit for stable landmark context (jawline/forehead).
    margin_x = int(box_w * 0.15)
    margin_y = int(box_h * 0.20)
    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(w, x_min + box_w + 2 * margin_x)
    y_max = min(h, y_min + box_h + 2 * margin_y)

    face_crop = image_array[y_min:y_max, x_min:x_max]
    if face_crop.size == 0 or face_crop.shape[0] < 32 or face_crop.shape[1] < 32:
        return None

    mobilefacenet_embedding = extract_mobilefacenet_embedding(face_crop)
    if mobilefacenet_embedding is not None:
        return mobilefacenet_embedding

    # Fallback path: keep FaceMesh landmark embedding available when model is missing.
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(face_crop)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
        eye_centroid = (landmarks[33] + landmarks[263]) / 2.0
        landmarks = landmarks - eye_centroid
        eye_distance = np.linalg.norm(landmarks[33] - landmarks[263])
        if eye_distance > 0:
            landmarks = landmarks / eye_distance

        return landmarks.flatten().astype(np.float32)


def get_mobilefacenet_session():
    """Lazily initialize ONNX Runtime session for MobileFaceNet."""
    global _mobilefacenet_session, _mobilefacenet_input_name, _mobilefacenet_output_name
    global _mobilefacenet_input_height, _mobilefacenet_input_width, _mobilefacenet_input_layout

    if _mobilefacenet_session is not None:
        return _mobilefacenet_session, _mobilefacenet_input_name, _mobilefacenet_output_name

    if not os.path.exists(MOBILEFACENET_MODEL_PATH):
        logger.warning(
            "MobileFaceNet model not found at %s. Falling back to FaceMesh embedding.",
            MOBILEFACENET_MODEL_PATH,
        )
        return None, None, None

    try:
        session = ort.InferenceSession(
            MOBILEFACENET_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        input_name = input_meta.name
        output_name = output_meta.name

        input_shape = input_meta.shape
        if len(input_shape) == 4:
            # Typical models are either NCHW [N,3,H,W] or NHWC [N,H,W,3].
            if input_shape[1] == 3:
                _mobilefacenet_input_layout = "NCHW"
                _mobilefacenet_input_height = int(input_shape[2]) if isinstance(input_shape[2], int) else 112
                _mobilefacenet_input_width = int(input_shape[3]) if isinstance(input_shape[3], int) else 112
            elif input_shape[3] == 3:
                _mobilefacenet_input_layout = "NHWC"
                _mobilefacenet_input_height = int(input_shape[1]) if isinstance(input_shape[1], int) else 112
                _mobilefacenet_input_width = int(input_shape[2]) if isinstance(input_shape[2], int) else 112

        _mobilefacenet_session = session
        _mobilefacenet_input_name = input_name
        _mobilefacenet_output_name = output_name
        logger.info(
            "Loaded MobileFaceNet model from %s (input=%s, layout=%s, resize=%dx%d)",
            MOBILEFACENET_MODEL_PATH,
            input_shape,
            _mobilefacenet_input_layout,
            _mobilefacenet_input_width,
            _mobilefacenet_input_height,
        )

        output_shape = output_meta.shape
        if isinstance(output_shape, list):
            known_embed_dims = {64, 96, 128, 192, 256, 384, 512}
            flat_dim = None
            numeric_dims = [d for d in output_shape if isinstance(d, int)]
            if numeric_dims:
                flat_dim = int(np.prod(numeric_dims))
            if flat_dim is not None and flat_dim not in known_embed_dims:
                logger.warning(
                    "Model output shape %s (flat=%d) is unusual for face embeddings; "
                    "you may be using a classification model.",
                    output_shape,
                    flat_dim,
                )
        return session, input_name, output_name
    except Exception as exc:
        logger.error("Failed to load MobileFaceNet model: %s", exc)
        return None, None, None


def preprocess_mobilefacenet(face_rgb: np.ndarray) -> np.ndarray:
    """Prepare RGB face crop for MobileFaceNet ONNX inference."""
    resized = cv2.resize(face_rgb, (_mobilefacenet_input_width, _mobilefacenet_input_height))
    normalized = (resized.astype(np.float32) - 127.5) / 128.0
    if _mobilefacenet_input_layout == "NHWC":
        return np.expand_dims(normalized, axis=0)
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def extract_mobilefacenet_embedding(face_crop: np.ndarray):
    """Run MobileFaceNet ONNX model and return L2-normalized embedding."""
    session, input_name, output_name = get_mobilefacenet_session()
    if session is None:
        return None

    try:
        model_input = preprocess_mobilefacenet(face_crop)
        outputs = session.run([output_name], {input_name: model_input})
        embedding = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            return None
        return embedding / norm
    except Exception as exc:
        logger.error("MobileFaceNet inference failed: %s", exc)
        return None


class SimHasher:
    """
    Locality-Sensitive Hashing (LSH) for face embeddings.
    Converts embeddings into a 256-bit binary hash (64-char hex string).
    """
    def __init__(self, num_bits=256, input_dim=128, seed=42):
        self.num_bits = num_bits
        self.input_dim = input_dim
        # Fixed random seed ensures the same hyperplanes across server restarts
        rng = np.random.default_rng(seed)
        self.planes = rng.standard_normal((self.num_bits, self.input_dim))
        
    def compute(self, embedding) -> str:
        """
        Compute the SimHash for a given FaceMesh embedding vector.
        Returns a 64-character hex string (256 bits).
        """
        emb_array = np.array(embedding)
        projections = self.planes @ emb_array
        bits = "".join("1" if p > 0 else "0" for p in projections)
        return f"{int(bits, 2):064x}"

def hamming_distance(h1: str, h2: str) -> int:
    """
    Compute the Hamming Distance between two 256-bit SimHash hex strings.
    """
    try:
        b1 = f"{int(h1, 16):0256b}"
        b2 = f"{int(h2, 16):0256b}"
        # Pad to the same length in case of mismatched inputs
        max_len = max(len(b1), len(b2))
        b1 = b1.zfill(max_len)
        b2 = b2.zfill(max_len)
        return sum(a != b for a, b in zip(b1, b2))
    except ValueError:
        return 256


def generate_face_hash(embedding) -> str:
    """
    Generate SimHash of face embedding for privacy-preserving verification.
    """
    emb_array = np.asarray(embedding).reshape(-1)
    hasher = SimHasher(input_dim=int(emb_array.shape[0]))
    return hasher.compute(emb_array)


def analyze_face_mesh(image_array):
    """
    Analyze face using MediaPipe Face Mesh (BONUS).

    TODO: Implement using:
    - mediapipe.solutions.face_mesh.FaceMesh
    - Extract 468 landmarks
    - Calculate depth from nose_tip_z (landmark index 1)
    - Check mesh completeness

    Return dict with:
    - face_mesh_complete: bool
    - landmark_count: int
    - nose_tip_z: float
    - depth_quality: "good" | "moderate" | "poor"
    """
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_array)
        
        if not results.multi_face_landmarks:
            return {
                "face_mesh_complete": False,
                "landmark_count": 0,
                "nose_tip_z": 0.0,
                "depth_quality": "poor"
            }
            
        face_landmarks = results.multi_face_landmarks[0]
        landmark_count = len(face_landmarks.landmark)
        
        # Check completeness (MediaPipe usually returns 468 landmarks without refinement)
        face_mesh_complete = landmark_count >= 468
        
        # Calculate depth from nose tip (landmark index 1)
        nose_tip_z = face_landmarks.landmark[1].z
        abs_z = abs(nose_tip_z)
        
        if abs_z > 0.03:
            depth_quality = "good"
        elif abs_z < 0.01:
            depth_quality = "poor"
        else:
            depth_quality = "moderate"
            
        return {
            "face_mesh_complete": face_mesh_complete,
            "landmark_count": landmark_count,
            "nose_tip_z": float(nose_tip_z),
            "depth_quality": depth_quality
        }


def detect_vpn_proxy(ip_address: str, user_agent: str) -> tuple:
    """
    Detect VPN/proxy usage.

    TODO: Check for:
    - Private IP ranges (10.x, 172.16-31.x, 192.168.x)
    - Localhost (127.x, ::1)
    - VPN keywords in user_agent
    - Known proxy headers (not available here, but could extend)

    Return (is_vpn: bool, confidence: float)
    """
    confidence = 0.0
    is_vpn = False

    ip = (ip_address or "").strip()
    ua = (user_agent or "").lower()

    if ip:
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved or ip_obj.is_link_local:
                is_vpn = True
                confidence = max(confidence, 0.7)
        except ValueError:
            # If IP is malformed, treat as suspicious network signal
            is_vpn = True
            confidence = max(confidence, 0.6)

    vpn_markers = ["vpn", "proxy", "tor", "wireguard", "openvpn", "tailscale", "cloudflare warp"]
    if any(marker in ua for marker in vpn_markers):
        is_vpn = True
        confidence = max(confidence, 0.9)

    return is_vpn, confidence


# =============================================================================
# PRIVACY REQUIREMENTS (IMPORTANT!)
# =============================================================================
"""
Your implementation MUST follow these privacy requirements:

1. NO RAW IMAGES STORED
   - Process images in-memory only
   - Do not write images to disk
   - Do not send images to external APIs

2. HASH-ONLY STORAGE
   - Store only SHA-256 hashes (64 hex characters)
   - Hashes are one-way - cannot reconstruct face
   - Different faces must produce different hashes

3. EPHEMERAL PROCESSING
   - Clear image data after processing
   - No caching of raw biometric data
   - Use Python's memory management (del, gc.collect)

4. CONSENT TRACKING
   - Require camera_consent=True for enrollment
   - Log consent in audit trail (backend responsibility)

5. RESPONSE HYGIENE
   - Never include base64 image data in responses
   - Only return hashes, scores, and metadata
"""
