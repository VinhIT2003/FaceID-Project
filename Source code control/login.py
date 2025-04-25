import cv2
import dlib
import time
import os
import platform
import socket
import numpy as np
import face_recognition

class FaceIDLogin:
    def __init__(self, username, sock):
        self.username = username.strip()
        self.sock = sock  # socket để gửi kết quả
        self.face_id_file = os.path.join("Face_ID", f"{self.username}.txt")

    def send_result(self, status, message):
        result = f"{status}:{self.username}:{message}"
        self.sock.send(result.encode())

    def login_with_face_id(self):
        if not os.path.exists(self.face_id_file):
            self.send_result("fail", "No Face ID registered")
            return

        try:
            with open(self.face_id_file, "rb") as f:
                stored_encoding = np.frombuffer(f.read(), dtype=np.float64)

            if stored_encoding.shape != (128,) or np.all(stored_encoding == 0):
                self.send_result("fail", "Invalid Face ID data")
                return

            # Camera initialization based on OS
            system_platform = platform.system()
            if system_platform == "Windows":
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(0)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                self.send_result("fail", "Cannot access camera")
                return

            detector = dlib.get_frontal_face_detector()
            predictor_path = "Face_ID_Project/shape_predictor_68_face_landmarks.dat"
            predictor = dlib.shape_predictor(predictor_path)

            def shape_to_np(shape):
                coords = np.zeros((68, 2), dtype=int)
                for i in range(68):
                    coords[i] = (shape.part(i).x, shape.part(i).y)
                return coords

            def calculate_ear(eye):
                A = np.linalg.norm(eye[1] - eye[5])
                B = np.linalg.norm(eye[2] - eye[4])
                C = np.linalg.norm(eye[0] - eye[3])
                return (A + B) / (2.0 * C)

            EAR_THRESHOLD = 0.25
            REQUIRED_BLINKS = 5
            SHARPNESS_THRESHOLD = 120
            FACE_MATCH_THRESHOLD = 0.44
            MIN_VERIFICATION_DURATION = 15
            MIN_EYE_CLOSED_FRAMES = 3
            MAX_NO_MOVEMENT = 20

            blink_count = 0
            eye_closed_frames = 0
            eye_blinking = False
            face_verified = False
            start_time = time.time()
            frame_counter = 0
            blink_timestamps = []
            no_movement_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.resize(frame, (640, 480))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_counter += 1

                face_locations = face_recognition.face_locations(rgb)
                if not face_locations:
                    no_movement_frames += 1
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                no_movement_frames = 0

                top, right, bottom, left = face_locations[0]
                face_crop = gray[top:bottom, left:right]
                sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var()
                if sharpness < SHARPNESS_THRESHOLD:
                    self.send_result("fail", "Blurry image detected")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                rects = detector(gray, 0)
                if rects:
                    shape = predictor(gray, rects[0])
                    shape = shape_to_np(shape)
                    leftEye = shape[42:48]
                    rightEye = shape[36:42]
                    ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0

                    if ear < EAR_THRESHOLD:
                        eye_closed_frames += 1
                        eye_blinking = True
                    else:
                        if eye_blinking:
                            if eye_closed_frames >= MIN_EYE_CLOSED_FRAMES:
                                now = time.time()
                                if len(blink_timestamps) == 0 or now - blink_timestamps[-1] > 1.0:
                                    blink_count += 1
                                    blink_timestamps.append(now)
                            eye_closed_frames = 0
                            eye_blinking = False

                if frame_counter % 5 == 0:
                    face_encodings = face_recognition.face_encodings(rgb, face_locations)
                    if face_encodings:
                        current_encoding = face_encodings[0]
                        distance = np.linalg.norm(current_encoding - stored_encoding)

                        if distance < FACE_MATCH_THRESHOLD:
                            face_verified = True
                        else:
                            self.send_result("fail", "Face ID does not match")
                            cap.release()
                            cv2.destroyAllWindows()
                            return

                        verification_duration = time.time() - start_time
                        if verification_duration >= MIN_VERIFICATION_DURATION:
                            if blink_count < REQUIRED_BLINKS or not face_verified or no_movement_frames > MAX_NO_MOVEMENT:
                                self.send_result("fail", "Verification failed - spoofing suspected")
                                cap.release()
                                cv2.destroyAllWindows()
                                return

                        if (
                            blink_count >= REQUIRED_BLINKS and
                            face_verified and
                            verification_duration >= MIN_VERIFICATION_DURATION
                        ):
                            self.send_result("success", "Face ID verified successfully")
                            cap.release()
                            cv2.destroyAllWindows()
                            return

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            self.send_result("fail", f"Error: {str(e)}")
