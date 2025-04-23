import cv2
import dlib
import time
import os
import subprocess
import numpy as np
from tkinter import messagebox
import face_recognition

class FaceIDLogin:
    def __init__(self, root, username_entry):
        self.root = root
        self.username = username_entry  # Entry chứa mã nhân viên

    def login_with_face_id(self):
        emp_no = self.username.get().strip()
        face_id_file = f"D:/Face_ID/{emp_no}.txt"

        if not os.path.exists(face_id_file):
            messagebox.showerror("Error", "You haven't registered your face. Please register your Face ID to log in", parent=self.root)
            return

        try:
            with open(face_id_file, "rb") as f:
                stored_encoding = np.frombuffer(f.read(), dtype=np.float64)

            if stored_encoding.shape != (128,) or np.all(stored_encoding == 0):
                messagebox.showerror("Error", "Invalid Face ID data!", parent=self.root)
                return

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                messagebox.showerror("Error", "Unable to access the camera!", parent=self.root)
                return

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("C:/Đồ án Python/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

            def shape_to_np(shape, dtype="int"):
                coords = np.zeros((68, 2), dtype=dtype)
                for i in range(68):
                    coords[i] = (shape.part(i).x, shape.part(i).y)
                return coords

            EAR_THRESHOLD = 0.25
            REQUIRED_BLINKS = 5
            SHARPNESS_THRESHOLD = 120
            FACE_MATCH_THRESHOLD = 0.44
            MIN_VERIFICATION_DURATION = 15

            blink_count = 0
            eye_closed_frames = 0
            eye_blinking = False
            face_verified = False
            start_time = time.time()
            frame_counter = 0
            blink_timestamps = []

            MIN_EYE_CLOSED_FRAMES = 3
            MAX_NO_MOVEMENT = 20
            no_movement_frames = 0

            def calculate_ear(eye):
                A = np.linalg.norm(eye[1] - eye[5])
                B = np.linalg.norm(eye[2] - eye[4])
                C = np.linalg.norm(eye[0] - eye[3])
                return (A + B) / (2.0 * C)

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
                    cv2.putText(frame, "No face detected!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Face ID Login", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                no_movement_frames = 0  # reset

                top, right, bottom, left = face_locations[0]
                face_crop = gray[top:bottom, left:right]
                sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var()

                if sharpness < SHARPNESS_THRESHOLD:
                    cap.release()
                    cv2.destroyAllWindows()
                    messagebox.showerror("Spoofing Alert", "Image is blurry – possible spoofing detected!", parent=self.root)
                    return

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

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
                            cap.release()
                            cv2.destroyAllWindows()
                            messagebox.showerror("Error", "Face ID does not match !", parent=self.root)
                            return

                        verification_duration = time.time() - start_time
                        if verification_duration >= MIN_VERIFICATION_DURATION:
                            if blink_count < REQUIRED_BLINKS or not face_verified or no_movement_frames > MAX_NO_MOVEMENT:
                                cap.release()
                                cv2.destroyAllWindows()
                                messagebox.showerror("Spoofing Alert", "Verification failed: possible spoofing detected!", parent=self.root)
                                return

                        if (
                            blink_count >= REQUIRED_BLINKS and
                            face_verified and
                            verification_duration >= MIN_VERIFICATION_DURATION
                        ):
                            cap.release()
                            cv2.destroyAllWindows()
                            messagebox.showinfo("Success", f"Welcome {emp_no}!", parent=self.root)
                            self.root.destroy()
                            subprocess.Popen(['python', 'dashboard.py', emp_no])
                            return

                cv2.putText(frame, f"Blink: {blink_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, "Verifying face...", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Face ID Login", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.root)
