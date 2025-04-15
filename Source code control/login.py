import cv2
import dlib
import numpy as np
import time
import face_recognition
import pyodbc
import subprocess
from tkinter import messagebox


class FaceIDLogin:
    def __init__(self, root):
        self.root = root
        self.predictor_path = "D:\Face_ID Project\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def calculate_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def login(self, emp_no):
        if not emp_no or emp_no == "Unknown":
            messagebox.showerror("Error", "Invalid employee number!", parent=self.root)
            return

        try:
            conn = pyodbc.connect(
                'DRIVER={ODBC Driver 17 for SQL Server};'
                'SERVER=LAPTOP-NGQUJ9MT\\QUANGVINH;'
                'DATABASE=Face_ID_Management;'
                'UID=sa;PWD=1234;'
            )
            cursor = conn.cursor()
            cursor.execute("SELECT Emp_No, Name, Face_ID FROM Employee WHERE Emp_No = ?", (emp_no,))
            row = cursor.fetchone()
            conn.close()

            if not row or not row[2]:
                messagebox.showerror("Error", "Employee has not registered Face ID!", parent=self.root)
                return

            emp_no, emp_name, face_id = row
            stored_encoding = np.frombuffer(face_id, dtype=np.float64)

            if stored_encoding.shape != (128,) or np.all(stored_encoding == 0):
                messagebox.showerror("Error", "Invalid Face ID or image printed!", parent=self.root)
                return

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                messagebox.showerror("Error", "Unable to access the camera!", parent=self.root)
                return

            EAR_THRESHOLD = 0.25
            REQUIRED_BLINKS = 3
            SHARPNESS_THRESHOLD = 200
            FACE_MATCH_THRESHOLD = 0.44
            MIN_VERIFICATION_DURATION = 15
            MIN_EYE_CLOSED_FRAMES = 3
            MAX_EYE_CLOSED_FRAMES = 10

            blink_count = 0
            eye_closed_frames = 0
            eye_blinking = False
            face_verified = False
            start_time = time.time()
            frame_counter = 0

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
                    cv2.putText(frame, "No face detected!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Face ID Login", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                top, right, bottom, left = face_locations[0]
                face_crop = gray[top:bottom, left:right]
                sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var()

                if sharpness < SHARPNESS_THRESHOLD:
                    cap.release()
                    cv2.destroyAllWindows()
                    messagebox.showerror("Spoofing Alert", "Image is too blurry – suspected spoofing attempt!", parent=self.root)
                    return

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                rects = self.detector(gray, 0)
                if rects:
                    shape = self.predictor(gray, rects[0])
                    shape = self.shape_to_np(shape)
                    leftEye = shape[42:48]
                    rightEye = shape[36:42]
                    ear = (self.calculate_ear(leftEye) + self.calculate_ear(rightEye)) / 2.0

                    if ear < EAR_THRESHOLD:
                        eye_closed_frames += 1
                        eye_blinking = True
                    else:
                        if eye_blinking:
                            if MIN_EYE_CLOSED_FRAMES <= eye_closed_frames <= MAX_EYE_CLOSED_FRAMES:
                                blink_count += 1
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
                            messagebox.showerror("Error", "Invalid Face ID or image printed!", parent=self.root)
                            return

                        verification_duration = time.time() - start_time

                        if verification_duration >= MIN_VERIFICATION_DURATION:
                            if blink_count < REQUIRED_BLINKS or not face_verified:
                                cap.release()
                                cv2.destroyAllWindows()
                                messagebox.showerror("Spoofing Alert", "Verification failed: suspected image spoofing!", parent=self.root)
                                return

                        if (
                            blink_count >= REQUIRED_BLINKS and
                            face_verified and
                            verification_duration >= MIN_VERIFICATION_DURATION
                        ):
                            cap.release()
                            cv2.destroyAllWindows()
                            messagebox.showinfo("Success", f"Welcome {emp_name}!", parent=self.root)
                            self.root.destroy()
                            subprocess.Popen(['python', 'dashboard.py', emp_no, emp_name])
                            return

                cv2.putText(frame, f"Blink: {blink_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, "Verifying face...", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow("Face ID Login", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

        except pyodbc.Error as e:
            messagebox.showerror("Database Error", str(e), parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.root)
