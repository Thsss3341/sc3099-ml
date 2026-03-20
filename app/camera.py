import cv2
import time
import sys
import os
import asyncio
import base64
import random
import numpy as np

# Add current directory to path so we can import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set Redis URL to match user's Docker config (port 6380)
os.environ["REDIS_URL"] = "redis://localhost:6380/0"

try:
    import main
except ImportError as e:
    print(f"Error: Could not import 'main.py'. Reason: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while importing 'main.py': {e}")
    sys.exit(1)

def encode_image_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

OVERLAY_ALPHA = 0.6

# Direction metadata for each challenge type
DIRECTION_META = {
    "head_left":  {"label": "Turn HEAD LEFT",  "instruction": "Turn your head to the LEFT and CAPTURE"},
    "head_right": {"label": "Turn HEAD RIGHT", "instruction": "Turn your head to the RIGHT and CAPTURE"},
    "head_up":    {"label": "Look UP",          "instruction": "Tilt your head UP and CAPTURE"},
    "head_down":  {"label": "Look DOWN",        "instruction": "Tilt your head DOWN and CAPTURE"},
}

def make_verify_steps():
    """Build a fresh randomised 3-step flow each time verification starts."""
    lr = random.choice(["head_left", "head_right"])
    ud = random.choice(["head_up", "head_down"])
    return [
        {"key": "face", "label": "Step 1/3: Face Match",
         "instruction": "Look straight at the camera, then CAPTURE"},
        {"key": lr, "label": f"Step 2/3: {DIRECTION_META[lr]['label']}",
         "instruction": DIRECTION_META[lr]["instruction"]},
        {"key": ud, "label": f"Step 3/3: {DIRECTION_META[ud]['label']}",
         "instruction": DIRECTION_META[ud]["instruction"]},
    ]


def draw_ui(frame, mode, response_msg, step_info=None):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Header banner
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    # Footer banner
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)

    # Mode text
    mode_color = (0, 255, 0) if mode == "ENROLLMENT" else (255, 165, 0)
    cv2.putText(frame, f"MODE: {mode}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)

    # Verification step banner (shows step label + instruction)
    text_y_start = 80
    if step_info is not None:
        cv2.putText(frame, step_info["label"], (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(frame, step_info["instruction"], (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)
        text_y_start = 130

    # Footer instructions
    cv2.putText(frame, "Press 'm' to switch mode | 'c'/SPACE to capture | 'q' quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Response message box
    if response_msg:
        lines = response_msg.split('\n')
        box_h = len(lines) * 33 + 20
        box_overlay = frame.copy()
        cv2.rectangle(box_overlay, (15, text_y_start - 10),
                      (625, text_y_start + box_h), (0, 0, 0), -1)
        cv2.addWeighted(box_overlay, 0.70, frame, 0.30, 0, frame)

        for i, line in enumerate(lines):
            color = (255, 255, 255)
            if "FAILED" in line or "Error" in line:
                color = (0, 80, 255)
            elif "PASSED" in line or "SUCCESS" in line or "✓" in line:
                color = (50, 220, 50)
            cv2.putText(frame, line, (30, text_y_start + 22 + i * 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

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

    # Verification state machine
    verify_step = None   # None = idle; 0/1/2 = current step index
    verify_steps = make_verify_steps()  # initialise with a default set
    verify_results = {}  # Accumulated results: {step_key: {passed, score, label}}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        display_frame = frame.copy()

        if shutter_frames > 0:
            display_frame[:] = 255
            shutter_frames -= 1
        else:
            if time.time() > msg_timer:
                response_msg = ""

            # Pass current step_info to the UI so the prompt shows
            cur_step_info = verify_steps[verify_step] if (mode == "VERIFICATION" and verify_step is not None) else None
            display_frame = draw_ui(display_frame, mode, response_msg, cur_step_info)

        cv2.imshow('Camera Test', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key == ord('m'):
            mode = "VERIFICATION" if mode == "ENROLLMENT" else "ENROLLMENT"
            # Reset verification state on mode switch
            verify_step = None
            verify_results = {}
            response_msg = f"Switched to {mode} mode"
            msg_timer = time.time() + 2.0
            print(response_msg)

        elif key == ord('c') or key == 32:  # 32 = SPACE
            shutter_frames = 3
            print(f"\nCapturing in {mode} mode...")

            response_msg = "Processing... Please wait."
            msg_timer = time.time() + 10.0
            cur_step_info = verify_steps[verify_step] if (mode == "VERIFICATION" and verify_step is not None) else None
            flash_frame = draw_ui(frame.copy(), mode, response_msg, cur_step_info)
            cv2.imshow('Camera Test', flash_frame)
            cv2.waitKey(1)

            b64_image = encode_image_to_base64(frame)

            # ── ENROLLMENT ───────────────────────────────────────────────────
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
                        response_msg = (f"Enrollment SUCCESS!\n"
                                        f"Quality: {res.quality_score:.2f}\n"
                                        f"Template: {reference_hash[:16]}...")
                    else:
                        error_msg = res.details.get('error', 'Unknown Error')
                        response_msg = (f"Enrollment FAILED!\n"
                                        f"Quality: {res.quality_score:.2f}\n"
                                        f"Reason: {error_msg}")
                except Exception as e:
                    response_msg = f"Enrollment Error:\n{e.detail if hasattr(e, 'detail') else str(e)}"

            # ── VERIFICATION (3-step flow) ────────────────────────────────────
            elif mode == "VERIFICATION":
                if not reference_hash:
                    response_msg = "Error: Please ENROLL first\nbefore verifying!"
                    verify_step = None
                else:
                    # Start the flow from step 0 if idle
                    if verify_step is None:
                        verify_steps = make_verify_steps()  # fresh random directions
                        verify_step = 0
                        verify_results = {}

                    step_info = verify_steps[verify_step]
                    step_key = step_info["key"]
                    print(f"  → {step_info['label']}")

                    try:
                        # --- Step 1: Face Match ---
                        if step_key == "face":
                            req = main.FaceVerifyRequest(
                                image=b64_image,
                                reference_template_hash=reference_hash
                            )
                            res = asyncio.run(main.verify_face(req))
                            verify_results["face"] = {
                                "passed": res.match_passed,
                                "score": res.match_score,
                                "label": "Face Match"
                            }
                            if res.match_passed:
                                next_label = verify_steps[1]["label"]
                                response_msg = (f"✓ Face Match PASSED (score: {res.match_score:.2f})\n"
                                                f"Ready for {next_label}")
                                verify_step += 1
                            else:
                                response_msg = (f"✗ Face Match FAILED (score: {res.match_score:.2f})\n"
                                                f"Verification aborted. Try enrolling again.")
                                verify_step = None
                                verify_results = {}

                        # --- Steps 2 & 3: Directional head challenges ---
                        else:
                            req = main.LivenessRequest(
                                challenge_response=b64_image,
                                challenge_type=step_key
                            )
                            res = asyncio.run(main.check_liveness(req))

                            # Get the relevant positional hint
                            nose_x = res.details.get("nose_x", None)
                            nose_y = res.details.get("nose_y", None)
                            if nose_x is not None:
                                hint = f" (nose_x={nose_x:.2f})"
                            elif nose_y is not None:
                                hint = f" (nose_y={nose_y:.2f})"
                            else:
                                hint = ""

                            verify_results[step_key] = {
                                "passed": res.liveness_passed,
                                "score": res.liveness_score,
                                "label": step_info["label"].split(": ", 1)[-1]  # strip "Step N/3: "
                            }
                            icon = "✓" if res.liveness_passed else "✗"
                            status = "PASSED" if res.liveness_passed else "FAILED"
                            response_msg = f"{icon} {step_info['label']} {status} (score: {res.liveness_score:.2f}){hint}"
                            verify_step += 1

                        # --- All steps done — final verdict ---
                        if verify_step is not None and verify_step == len(verify_steps):
                            all_passed = all(v["passed"] for v in verify_results.values())
                            avg_score = sum(v["score"] for v in verify_results.values()) / len(verify_results)
                            lines = ["=== Verification Complete ==="]
                            for v in verify_results.values():
                                icon = "✓" if v["passed"] else "✗"
                                lines.append(f"{icon} {v['label']}: {v['score']:.2f}")
                            overall = "PASSED" if all_passed else "FAILED"
                            lines.append(f"Overall: {overall}  (avg score: {avg_score:.2f})")
                            response_msg = "\n".join(lines)
                            print(response_msg.replace('\n', ' | '))
                            verify_step = None
                            verify_results = {}

                    except Exception as e:
                        err_detail = e.detail if hasattr(e, 'detail') else str(e)
                        response_msg = f"Error at {step_info['label']}:\n{err_detail}"
                        verify_step = None
                        verify_results = {}

            msg_timer = time.time() + 8.0
            print(response_msg.replace('\n', ' | '))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        cv2.imshow("Test", layer=None)
    except Exception:
        pass

    test_camera()
