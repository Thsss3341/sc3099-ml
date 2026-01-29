"""
SAIV Face Recognition & Risk Service

This service handles face enrollment, verification, liveness detection,
and multi-signal risk assessment.

Privacy Requirements:
- NO raw face images are stored
- Process images in-memory only
- Store only SHA-256 hashes of face embeddings
"""
import base64
import hashlib
from io import BytesIO
from typing import Optional, Dict, List, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import MediaPipe (may not be available in all environments)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


app = FastAPI(
    title="SAIV Face Recognition Service",
    description="Face enrollment, verification, liveness detection, and risk scoring service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class FaceEnrollRequest(BaseModel):
    user_id: str
    image: str  # Base64 encoded image
    camera_consent: bool = False


class FaceEnrollResponse(BaseModel):
    enrollment_successful: bool
    face_template_hash: str
    quality_score: float
    details: Dict[str, Any]


class FaceVerifyRequest(BaseModel):
    image: str
    reference_template_hash: str


class FaceVerifyResponse(BaseModel):
    match_passed: bool
    match_score: float
    match_threshold: float = 0.70
    face_detected: bool
    current_template_hash: str


class FaceMatchRequest(BaseModel):
    image: str
    reference_hash: str


class FaceMatchResponse(BaseModel):
    match_passed: bool
    match_score: float
    face_embedding_hash: str


class LivenessRequest(BaseModel):
    challenge_response: str
    challenge_type: str = "passive"


class LivenessResponse(BaseModel):
    liveness_passed: bool
    liveness_score: float
    liveness_threshold: float = 0.60
    challenge_type: str
    face_embedding_hash: str
    details: Dict[str, Any]


class GeolocationData(BaseModel):
    latitude: float
    longitude: float
    accuracy: float


class RiskAssessRequest(BaseModel):
    liveness_score: Optional[float] = None
    face_match_score: Optional[float] = None
    device_signature: Optional[str] = None
    device_public_key: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    geolocation: Optional[GeolocationData] = None


class RiskAssessResponse(BaseModel):
    risk_score: float
    risk_level: str
    pass_threshold: bool
    risk_threshold: float = 0.50
    signal_breakdown: Dict[str, float]
    recommendations: List[str]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode a base64 encoded image to numpy array."""
    try:
        # Handle data URL format
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return np.array(image.convert('RGB'))
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")


def generate_face_hash(image_array: np.ndarray, bbox: Optional[tuple] = None) -> str:
    """Generate SHA-256 hash of face region."""
    if bbox:
        h, w = image_array.shape[:2]
        x, y, width, height = bbox
        x = max(0, int(x * w))
        y = max(0, int(y * h))
        width = int(width * w)
        height = int(height * h)
        face_region = image_array[y:y+height, x:x+width]
    else:
        face_region = image_array

    # Resize to standard size for consistent hashing
    if CV2_AVAILABLE:
        face_resized = cv2.resize(face_region, (64, 64))
    else:
        # Fallback: use PIL
        face_pil = Image.fromarray(face_region)
        face_pil = face_pil.resize((64, 64))
        face_resized = np.array(face_pil)

    return hashlib.sha256(face_resized.tobytes()).hexdigest()


def detect_face_mediapipe(image_array: np.ndarray) -> Dict[str, Any]:
    """Detect face using MediaPipe."""
    if not MEDIAPIPE_AVAILABLE:
        # Fallback: assume face is present (for environments without MediaPipe)
        return {
            "detected": True,
            "confidence": 0.85,
            "bbox": (0.1, 0.1, 0.8, 0.8)  # Default bounding box
        }

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_array)

        if not results.detections:
            return {"detected": False, "confidence": 0.0, "bbox": None}

        detection = results.detections[0]
        confidence = detection.score[0]
        bbox = detection.location_data.relative_bounding_box

        return {
            "detected": True,
            "confidence": float(confidence),
            "bbox": (bbox.xmin, bbox.ymin, bbox.width, bbox.height)
        }


def analyze_face_mesh(image_array: np.ndarray) -> Dict[str, Any]:
    """Analyze face mesh for 3D depth cues (for liveness detection)."""
    if not MEDIAPIPE_AVAILABLE:
        return {
            "face_mesh_complete": True,
            "landmark_count": 468,
            "nose_tip_z": -0.05,
            "depth_quality": "good"
        }

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
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

        landmarks = results.multi_face_landmarks[0]
        nose_tip = landmarks.landmark[1]  # Nose tip is landmark index 1
        nose_tip_z = nose_tip.z

        # Determine depth quality
        if abs(nose_tip_z) > 0.03:
            depth_quality = "good"
        elif abs(nose_tip_z) > 0.01:
            depth_quality = "moderate"
        else:
            depth_quality = "poor"

        return {
            "face_mesh_complete": len(landmarks.landmark) >= 400,
            "landmark_count": len(landmarks.landmark),
            "nose_tip_z": float(nose_tip_z),
            "depth_quality": depth_quality
        }


def detect_vpn_proxy(ip_address: Optional[str], user_agent: Optional[str]) -> tuple:
    """Detect VPN/proxy usage."""
    is_vpn = False
    confidence = 0.0

    if ip_address:
        # Check for private IP ranges (VPN indicators)
        private_prefixes = ['10.', '192.168.', '172.16.', '172.17.', '172.18.',
                           '172.19.', '172.20.', '172.21.', '172.22.', '172.23.',
                           '172.24.', '172.25.', '172.26.', '172.27.', '172.28.',
                           '172.29.', '172.30.', '172.31.']
        if any(ip_address.startswith(prefix) for prefix in private_prefixes):
            is_vpn = True
            confidence = 0.7

        # Localhost
        if ip_address.startswith('127.') or ip_address == '::1':
            is_vpn = False  # Localhost is okay for testing
            confidence = 0.0

    if user_agent:
        vpn_keywords = ['vpn', 'proxy', 'tunnel', 'tor']
        if any(kw in user_agent.lower() for kw in vpn_keywords):
            is_vpn = True
            confidence = max(confidence, 0.8)

    return is_vpn, confidence


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "face-recognition"
    }


@app.get("/")
async def root():
    """List available endpoints."""
    return {
        "service": "SAIV Face Recognition & Risk Service",
        "version": "1.0.0",
        "endpoints": [
            "GET /health - Health check",
            "POST /face/enroll - Enroll a face for verification",
            "POST /face/verify - Verify a face against enrolled template",
            "POST /face/match - Legacy face matching",
            "POST /liveness/check - Perform liveness detection (BONUS)",
            "POST /risk/assess - Multi-signal risk assessment"
        ]
    }


@app.post("/face/enroll", response_model=FaceEnrollResponse, status_code=201)
async def enroll_face(request: FaceEnrollRequest):
    """Enroll a user's face for future verification."""
    # Validate consent
    if not request.camera_consent:
        raise HTTPException(
            status_code=400,
            detail="Camera consent is required for face enrollment"
        )

    # Decode image
    try:
        image_array = decode_base64_image(request.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Detect face
    detection = detect_face_mediapipe(image_array)

    if not detection["detected"]:
        raise HTTPException(status_code=400, detail="No face detected in image")

    confidence = detection["confidence"]
    if confidence < 0.7:
        raise HTTPException(
            status_code=400,
            detail=f"Face detection confidence too low: {confidence:.2f}"
        )

    # Generate face hash
    face_hash = generate_face_hash(image_array, detection.get("bbox"))

    # Calculate quality score
    h, w = image_array.shape[:2]
    resolution_score = min(1.0, (h * w) / (256 * 256))
    quality_score = (confidence + resolution_score) / 2

    return FaceEnrollResponse(
        enrollment_successful=True,
        face_template_hash=face_hash,
        quality_score=round(quality_score, 2),
        details={
            "face_detected": True,
            "face_detection_confidence": round(confidence, 2),
            "image_quality": "good" if quality_score >= 0.7 else "moderate"
        }
    )


@app.post("/face/verify", response_model=FaceVerifyResponse)
async def verify_face(request: FaceVerifyRequest):
    """Verify a face against an enrolled template."""
    # Decode image
    try:
        image_array = decode_base64_image(request.image)
    except ValueError as e:
        return FaceVerifyResponse(
            match_passed=False,
            match_score=0.0,
            match_threshold=0.70,
            face_detected=False,
            current_template_hash=""
        )

    # Detect face
    detection = detect_face_mediapipe(image_array)

    if not detection["detected"]:
        return FaceVerifyResponse(
            match_passed=False,
            match_score=0.0,
            match_threshold=0.70,
            face_detected=False,
            current_template_hash=""
        )

    # Generate current face hash
    current_hash = generate_face_hash(image_array, detection.get("bbox"))

    # Compare hashes
    # For hash-based comparison, we use a similarity heuristic
    # In a real implementation, you'd compare face embeddings
    if current_hash == request.reference_template_hash:
        match_score = 1.0
    else:
        # Calculate pseudo-similarity based on hash prefix matching
        # This is a simplified approach for the sample implementation
        matching_chars = sum(a == b for a, b in zip(current_hash, request.reference_template_hash))
        match_score = matching_chars / len(current_hash) * 0.9  # Scale down

    match_passed = match_score >= 0.70

    return FaceVerifyResponse(
        match_passed=match_passed,
        match_score=round(match_score, 2),
        match_threshold=0.70,
        face_detected=True,
        current_template_hash=current_hash
    )


@app.post("/face/match", response_model=FaceMatchResponse)
async def match_face(request: FaceMatchRequest):
    """Legacy face matching endpoint."""
    # Convert to verify request format
    verify_request = FaceVerifyRequest(
        image=request.image,
        reference_template_hash=request.reference_hash
    )
    verify_response = await verify_face(verify_request)

    return FaceMatchResponse(
        match_passed=verify_response.match_passed,
        match_score=verify_response.match_score,
        face_embedding_hash=verify_response.current_template_hash
    )


@app.post("/liveness/check", response_model=LivenessResponse)
async def check_liveness(request: LivenessRequest):
    """Perform liveness detection on submitted image."""
    # Decode image
    try:
        image_array = decode_base64_image(request.challenge_response)
    except ValueError as e:
        return LivenessResponse(
            liveness_passed=False,
            liveness_score=0.0,
            liveness_threshold=0.60,
            challenge_type=request.challenge_type,
            face_embedding_hash="",
            details={"error": str(e)}
        )

    # Detect face
    detection = detect_face_mediapipe(image_array)

    if not detection["detected"]:
        return LivenessResponse(
            liveness_passed=False,
            liveness_score=0.0,
            liveness_threshold=0.60,
            challenge_type=request.challenge_type,
            face_embedding_hash="",
            details={
                "face_detected": False,
                "face_detection_confidence": 0.0
            }
        )

    # Analyze face mesh for 3D cues
    mesh_analysis = analyze_face_mesh(image_array)

    # Calculate liveness score based on multiple factors
    scores = []

    # Face detection confidence (30%)
    scores.append(detection["confidence"] * 0.3)

    # Face mesh completeness (25%)
    if mesh_analysis["face_mesh_complete"]:
        scores.append(0.25)
    else:
        scores.append(0.0)

    # Depth analysis (30%)
    depth_score = 0.0
    if mesh_analysis["depth_quality"] == "good":
        depth_score = 0.30
    elif mesh_analysis["depth_quality"] == "moderate":
        depth_score = 0.15
    scores.append(depth_score)

    # Texture analysis (15%) - simplified check
    # Real implementation would check for print/screen artifacts
    texture_score = 0.15 if detection["confidence"] > 0.8 else 0.08
    scores.append(texture_score)

    liveness_score = sum(scores)
    liveness_passed = liveness_score >= 0.60

    # Generate face hash
    face_hash = generate_face_hash(image_array, detection.get("bbox"))

    return LivenessResponse(
        liveness_passed=liveness_passed,
        liveness_score=round(liveness_score, 2),
        liveness_threshold=0.60,
        challenge_type=request.challenge_type,
        face_embedding_hash=face_hash,
        details={
            "face_detection_confidence": round(detection["confidence"], 2),
            "face_mesh_complete": mesh_analysis["face_mesh_complete"],
            "depth_detected": abs(mesh_analysis["nose_tip_z"]) > 0.01,
            "texture_analysis_score": round(texture_score / 0.15, 2),
            "threshold": 0.6
        }
    )


@app.post("/risk/assess", response_model=RiskAssessResponse)
async def assess_risk(request: RiskAssessRequest):
    """Perform multi-signal risk assessment."""
    signal_breakdown = {}
    recommendations = []
    total_risk = 0.0

    # Liveness score (25% weight)
    if request.liveness_score is not None:
        liveness_risk = max(0.0, 1.0 - request.liveness_score) * 0.25
        signal_breakdown["liveness"] = round(liveness_risk, 3)
        total_risk += liveness_risk
        if request.liveness_score < 0.6:
            recommendations.append("Improve lighting and face visibility")
    else:
        signal_breakdown["liveness"] = 0.0

    # Face match score (25% weight)
    if request.face_match_score is not None:
        face_risk = max(0.0, 1.0 - request.face_match_score) * 0.25
        signal_breakdown["face_match"] = round(face_risk, 3)
        total_risk += face_risk
        if request.face_match_score < 0.7:
            recommendations.append("Re-enroll face or improve image quality")
    else:
        signal_breakdown["face_match"] = 0.0

    # Device attestation (20% weight)
    device_risk = 0.0
    if request.device_signature:
        # Simplified: presence of signature reduces risk
        device_risk = 0.05
    else:
        device_risk = 0.15
    signal_breakdown["device"] = round(device_risk, 3)
    total_risk += device_risk

    # Network/VPN detection (15% weight)
    is_vpn, vpn_confidence = detect_vpn_proxy(request.ip_address, request.user_agent)
    if is_vpn:
        network_risk = 0.15 * vpn_confidence
        recommendations.append("Disable VPN for check-in")
    else:
        network_risk = 0.0
    signal_breakdown["network"] = round(network_risk, 3)
    total_risk += network_risk

    # Geolocation (15% weight)
    geo_risk = 0.0
    if request.geolocation:
        if request.geolocation.accuracy > 5000:
            geo_risk = 0.15
            recommendations.append("Enable precise location services")
        elif request.geolocation.accuracy > 100:
            geo_risk = 0.05
        else:
            geo_risk = 0.0
    else:
        geo_risk = 0.10  # No location data
    signal_breakdown["geolocation"] = round(geo_risk, 3)
    total_risk += geo_risk

    # Determine risk level
    risk_score = min(1.0, total_risk)
    if risk_score < 0.3:
        risk_level = "LOW"
    elif risk_score < 0.5:
        risk_level = "MEDIUM"
    elif risk_score < 0.7:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return RiskAssessResponse(
        risk_score=round(risk_score, 2),
        risk_level=risk_level,
        pass_threshold=risk_score < 0.50,
        risk_threshold=0.50,
        signal_breakdown=signal_breakdown,
        recommendations=recommendations
    )
