import cv2
import time
import sys
import os
import asyncio
import base64
import numpy as np

# Add current directory to path so we can import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set Redis URL to match user's Docker config (port 6380)
os.environ["REDIS_URL"] = "redis://localhost:6380/0"

try:
    import main_test as main
except ImportError as e:
    print(f"Error: Could not import 'main.py'. Reason: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while importing 'main.py': {e}")
    sys.exit(1)

def encode_image_to_base64(frame):
    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Overlay parameters
OVERLAY_ALPHA = 0.6

def draw_ui(frame, mode, response_msg):
    # Create an overlay for UI
    overlay = frame.copy()
    
    # Draw bottom banner
    h, w = frame.shape[:2]
    # Header banner
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    # Footer banner
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
    
    # Mode text
    mode_color = (0, 255, 0) if mode == "ENROLLMENT" else (255, 165, 0)
    cv2.putText(frame, f"MODE: {mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)
    
    # Instructions
    cv2.putText(frame, "Press 'm' to switch mode | 'c' or SPACE to capture | 'q' to quit", (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Response Box
    if response_msg:
        # Draw semi-transparent box for text
        lines = response_msg.split('\n')
        box_h = len(lines) * 35 + 20
        box_overlay = frame.copy()
        cv2.rectangle(box_overlay, (20, 80), (600, 80 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(box_overlay, 0.7, frame, 0.3, 0, frame)
        
        y0 = 110
        for i, line in enumerate(lines):
            text_color = (0, 0, 255) if "Error" in line or "FAILED" in line else (255, 255, 255)
            if "SUCCESS" in line or "PASSED" in line:
                text_color = (0, 255, 0)
            cv2.putText(frame, line, (40, y0 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    return frame

def test_camera():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully.")
    
    mode = "ENROLLMENT"
    reference_hash = None
    response_msg = ""
    msg_timer = 0
    shutter_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
            
        display_frame = frame.copy()
        
        if shutter_frames > 0:
            # Shutter flash effect (white screen)
            display_frame[:] = 255
            shutter_frames -= 1
        else:
            if time.time() > msg_timer:
                response_msg = ""
            display_frame = draw_ui(display_frame, mode, response_msg)

        cv2.imshow('Camera Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = "VERIFICATION" if mode == "ENROLLMENT" else "ENROLLMENT"
            response_msg = f"Switched to {mode} mode"
            msg_timer = time.time() + 2.0
            print(response_msg)
        elif key == ord('c') or key == 32: # 32 is space
            shutter_frames = 3
            print(f"\nCapturing photo in {mode} mode...")
            
            # Show a temporary processing message
            response_msg = "Processing... Please wait."
            msg_timer = time.time() + 10.0 # Will be overwritten when done
            # Draw immediately so user sees "Processing"
            flash_frame = draw_ui(frame.copy(), mode, response_msg)
            cv2.imshow('Camera Test', flash_frame)
            cv2.waitKey(1)
            
            b64_image = encode_image_to_base64(frame)
            
            if mode == "ENROLLMENT":
                req = main.FaceEnrollRequest(
                    user_id="test_user",
                    image=b64_image,
                    camera_consent=True
                )
                try:
                    res = asyncio.run(main.enroll_face(req))
                    if res.enrollment_successful:
                        reference_hash = res.face_template_hash
                        response_msg = f"Enrollment SUCCESS!\nQuality: {res.quality_score:.2f}\nTemplate Hash: {reference_hash[:16]}..."
                    else:
                        error_msg = res.details.get('error', 'Unknown Error')
                        response_msg = f"Enrollment FAILED!\nQuality: {res.quality_score:.2f}\nReason: {error_msg}"
                except Exception as e:
                    response_msg = f"Error during enrollment:\n{str(e)}"
            
            elif mode == "VERIFICATION":
                if not reference_hash:
                    response_msg = "Error: Please ENROLL a face first\nbefore testing verification!"
                else:
                    req = main.FaceVerifyRequest(
                        image=b64_image,
                        user_id="test_user",
                        reference_simhash=reference_hash
                    )
                    try:
                        res = asyncio.run(main.verify_face(req))
                        dist_str = f"Hamming Dist: {res.hamming_dist}/128" if res.hamming_dist is not None else ""
                        if res.match_passed:
                            response_msg = f"Verification PASSED!\nMatch Score: {res.match_score:.2f} | {dist_str}\nUser: test_user"
                        else:
                            response_msg = f"Verification FAILED!\nMatch Score: {res.match_score:.2f} | {dist_str}"
                    except Exception as e:
                        response_msg = f"Error during verification:\n{str(e)}"
            
            msg_timer = time.time() + 6.0
            print(response_msg.replace('\n', ' - '))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check for headless opencv
    try:
        cv2.imshow("Test", layer=None)
    except Exception as e:
        pass

    test_camera()
