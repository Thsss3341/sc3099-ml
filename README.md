# SAIV Face Recognition & Risk Service (Module 3)

This service provides facial analysis, biometric verification, and risk assessment capabilities. It is designed to be consumed by the **Module 2 Backend API**.

**Base URL**: `http://face-recognition:8001` (Internal Docker Network)

## Privacy & Security Compliance
- **No Raw Images**: This service processes images in-memory and returns SHA-256 hashes. It does **not** persist images.
- **Hash Storage**: The Backend (Module 2) is responsible for storing the returned `face_template_hash` in PostgreSQL.
- **Similarity Matching**: This service internally uses Redis to cache FaceMesh vectors for functional similarity matching, while falling back to exact hash matching for strict privacy tests.

---

## API Endpoints

### 1. Enroll Face
Registers a user's face for future verification.

- **Endpoint**: `POST /face/enroll`
- **Purpose**: processing the initial enrollment photo.
- **Prerequisite**: User must have granted camera consent.

**Request Body**:
```json
{
  "user_id": "string (UUID)",
  "image": "string (Base64 encoded RGB image)",
  "camera_consent": true
}
```

**Response Body**:
```json
{
  "enrollment_successful": true,
  "face_template_hash": "string (64-char SHA-256 hex)",
  "quality_score": 0.95,
  "details": {
    "info": "Face enrolled successfully"
  }
}
```
> **Backend Integration Note**: Store `face_template_hash` in the `users` table.

---

### 2. Verify Face
Verifies a live face against a stored enrollment hash.

- **Endpoint**: `POST /face/verify`
- **Purpose**: Authenticating a user during check-in.

**Request Body**:
```json
{
  "image": "string (Base64 encoded RGB image)",
  "reference_template_hash": "string (The hash stored in users DB)"
}
```

**Response Body**:
```json
{
  "match_passed": true,
  "match_score": 0.88,
  "match_threshold": 0.85,
  "face_detected": true,
  "current_template_hash": "string (Hash of the incoming image)"
}
```
> **Logic**: Returns `match_passed: true` if `match_score >= match_threshold`.

---

### 3. Liveness Check
Detects if the submitted face is real or a spoof (photo/screen/mask).

- **Endpoint**: `POST /liveness/check`
- **Purpose**: Anti-spoofing verification.

**Request Body**:
```json
{
  "challenge_response": "string (Base64 encoded RGB image)",
  "challenge_type": "blink" 
}
```
*Supported types: "blink", "head_turn", "passive"*

**Response Body**:
```json
{
  "liveness_passed": true,
  "liveness_score": 0.92,
  "liveness_threshold": 0.60,
  "face_embedding_hash": "string (SHA-256)",
  "details": {
    "method": "quality_heuristic"
  }
}
```

---

### 4. Risk Assessment
Calculates a composite risk score based on multiple signals.

- **Endpoint**: `POST /risk/assess`
- **Purpose**: Determining if a check-in should be approved, flagged, or rejected.

**Request Body**:
```json
{
  "liveness_score": 0.9,
  "face_match_score": 0.88,
  "ip_address": "192.168.1.50",
  "user_agent": "Mozilla/5.0 ...",
  "geolocation": {
    "latitude": 1.3483,
    "longitude": 103.6831,
    "accuracy": 15.0
  }
}
```

**Response Body**:
```json
{
  "risk_score": 0.15,
  "risk_level": "LOW", 
  "pass_threshold": true,
  "risk_threshold": 0.5,
  "signal_breakdown": {
    "liveness": 0.1,
    "face_match": 0.12,
    "network": 0.0
  },
  "recommendations": []
}
```
*Risk Levels: LOW, MEDIUM, HIGH, CRITICAL*

---

### 5. Health Check
- **Endpoint**: `GET /health`
- **Response**: `{"status": "healthy", "service": "...", "redis": "connected"}`

## Error Handling
All endpoints return standard HTTP error codes:
- **422 Validation Error**: Invalid JSON or missing fields.
- **500 Internal Server Error**: Service failure.
