import cv2
import dlib
import numpy as np
import face_recognition
from imutils import face_utils
import os
import ctypes
import threading
from tkinter import messagebox

class FaceID_Register:
    def __init__(self, root, emp_no):
        self.root = root
        self.emp_no = emp_no

    def encode_face(self, rgb_frame, face_locations, cropped_face=None):
        """Mã hóa khuôn mặt"""
        try:
            if cropped_face is not None and cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                face_crop_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                cropped_encoding = face_recognition.face_encodings(face_crop_rgb)
                if cropped_encoding:
                    return np.array(cropped_encoding[0], dtype=np.float64).tobytes()

            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]])
                if face_encodings:
                    return np.array(face_encodings[0], dtype=np.float64).tobytes()

        except Exception as e:
            print(f"[ERROR] Lỗi mã hóa khuôn mặt: {e}")
        return None

    def register_face_id(self):
        if not self.emp_no or self.emp_no == "Unknown":
            messagebox.showerror("Error", "Cannot register face without an employee ID!", parent=self.root)
            return

        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                messagebox.showerror("Error", "Cannot open the camera!", parent=self.root)
                return

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("C:/Đồ án Python/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

            cv2.namedWindow("Face Scanning")

            prev_face_location = None
            blink_count = 0
            eye_closed_frames = 0

            sharpness_threshold = 100  
            light_change_threshold = 10  
            movement_threshold = 20  
            min_closed_frames = 5  
            required_blinks = 5  
            max_no_movement = 15  
            max_no_light_change = 30  

            no_movement_frames = 0
            no_light_change_frames = 0
            prev_light_intensity = None

            def calculate_ear(eye):
                A = np.linalg.norm(eye[1] - eye[5])
                B = np.linalg.norm(eye[2] - eye[4])
                C = np.linalg.norm(eye[0] - eye[3])
                return (A + B) / (2.0 * C)

            def stop_face_scan():
                cap.release()
                cv2.destroyAllWindows()

            def detect_liveness():
                nonlocal blink_count, prev_face_location, eye_closed_frames
                nonlocal no_movement_frames, prev_light_intensity, no_light_change_frames

                ret, frame = cap.read()
                if not ret:
                    return

                frame = cv2.resize(frame, (640, 480))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_frame, model="hog")

                if not face_locations:
                    cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Face Scanning", frame)
                    self.root.after(10, detect_liveness)
                    return

                if len(face_locations) > 1:
                    cv2.putText(frame, "Multiple faces detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Face Scanning", frame)
                    messagebox.showerror("Error", "Detected more than one face. Please ensure only one face is in view!", parent=self.root)
                    stop_face_scan()
                    return

                top, right, bottom, left = face_locations[0]
                offset = 25
                top = max(0, top - offset)
                bottom = min(frame.shape[0], bottom + offset)
                left = max(0, left - offset)
                right = min(frame.shape[1], right + offset)

                face_crop = frame[top:bottom, left:right]
                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    cv2.imshow("Cropped Face", face_crop)
                cv2.rectangle(frame, (left, top), (right, bottom), (34, 139, 34), 2)

                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                cv2.putText(frame, f"Sharpness: {int(sharpness)}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if sharpness < sharpness_threshold:
                    messagebox.showerror("Warning", "Detected a printed photo! Please use a real face.", parent=self.root)
                    stop_face_scan()
                    return

                light_intensity = np.mean(gray[top:bottom, left:right])
                if prev_light_intensity is not None:
                    light_diff = abs(light_intensity - prev_light_intensity)
                    no_light_change_frames = no_light_change_frames + 1 if light_diff < light_change_threshold else 0
                    if no_light_change_frames > max_no_light_change:
                        messagebox.showerror("Warning", "No light change detected! Possibly a static image.", parent=self.root)
                        stop_face_scan()
                        return
                prev_light_intensity = light_intensity

                if prev_face_location is not None:
                    dx = abs(prev_face_location[3] - left) + abs(prev_face_location[1] - right)
                    dy = abs(prev_face_location[0] - top) + abs(prev_face_location[2] - bottom)
                    no_movement_frames = no_movement_frames + 1 if dx < movement_threshold and dy < movement_threshold else 0
                prev_face_location = face_locations[0]

                if no_movement_frames >= max_no_movement:
                    messagebox.showerror("Warning", "Detected a printed photo due to lack of movement!", parent=self.root)
                    stop_face_scan()
                    return

                rects = detector(gray, 0)
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    leftEye = shape[36:42]
                    rightEye = shape[42:48]
                    leftEAR = calculate_ear(leftEye)
                    rightEAR = calculate_ear(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0

                    if ear < 0.22:
                        eye_closed_frames += 1
                    else:
                        if eye_closed_frames >= min_closed_frames:
                            if sharpness >= sharpness_threshold and no_movement_frames < max_no_movement:
                                blink_count += 1
                            eye_closed_frames = 0

                cv2.putText(frame, f"Blinks: {blink_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if blink_count >= required_blinks:
                    cv2.putText(frame, "Valid face!", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Face Scanning", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    if blink_count < required_blinks:
                        messagebox.showwarning("Warning", "Please blink more to verify!", parent=self.root)
                    else:
                        image_path = f"face_data/{self.emp_no}.png"
                        cv2.imwrite(image_path, face_crop)
                        print(f"[INFO] Face image saved: {image_path}")

                        face_encoding = self.encode_face(rgb_frame, face_locations, face_crop)
                        if face_encoding:
                            folder_path = "D:/Face_ID"
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)
                                ctypes.windll.kernel32.SetFileAttributesW(folder_path, 0x02)

                            file_path = os.path.join(folder_path, f"{self.emp_no}.txt")
                            with open(file_path, "wb") as f:
                                f.write(face_encoding)

                            messagebox.showinfo("Success", f"Face encoding saved at: {file_path}", parent=self.root)
                            stop_face_scan()
                            return
                        else:
                            messagebox.showerror("Error", "Failed to encode face!", parent=self.root)
                elif key == ord("q"):
                    stop_face_scan()
                    return

                self.root.after(10, detect_liveness)

            detect_liveness()
            self.root.protocol("WM_DELETE_WINDOW", stop_face_scan)

        except Exception as e:
            messagebox.showerror("Error", f"System error: {str(e)}", parent=self.root)
