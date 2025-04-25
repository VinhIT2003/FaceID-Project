import cv2
import dlib
import numpy as np
import face_recognition
from imutils import face_utils
import os
import ctypes
import platform
import json 

class FaceID_Register:
    def __init__(self, emp_no, conn):  # conn là socket đã accept
        self.emp_no = emp_no
        self.conn = conn  # socket connection từ server

    def send_result(self, status, message, image_path=None, encoding_path=None):
        result = {
            "status": status,
            "message": message,
            "image_path": image_path,
            "encoding_path": encoding_path
        }
        self.conn.sendall(json.dumps(result).encode())

    def encode_face(self, rgb_frame, face_locations, cropped_face=None):
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
            self.send_result("error", f"Encoding error: {e}")
        return None

    def register_face_id(self):
        if not self.emp_no or self.emp_no == "Unknown":
            self.send_result("error", "Invalid employee ID.")
            return

        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.send_result("error", "Cannot open the camera.")
                return

            detector = dlib.get_frontal_face_detector()
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                self.send_result("error", "Missing predictor model file.")
                return
            predictor = dlib.shape_predictor(predictor_path)

            # Thresholds
            sharpness_threshold = 100
            light_change_threshold = 10
            movement_threshold = 20
            min_closed_frames = 5
            required_blinks = 5
            max_no_movement = 15
            max_no_light_change = 30

            blink_count = 0
            eye_closed_frames = 0
            prev_face_location = None
            no_movement_frames = 0
            no_light_change_frames = 0
            prev_light_intensity = None

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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                if len(face_locations) != 1:
                    continue

                top, right, bottom, left = face_locations[0]
                offset = 25
                top = max(0, top - offset)
                bottom = min(frame.shape[0], bottom + offset)
                left = max(0, left - offset)
                right = min(frame.shape[1], right + offset)
                face_crop = frame[top:bottom, left:right]

                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness < sharpness_threshold:
                    self.send_result("error", "Photo detected (low sharpness).")
                    break

                light_intensity = np.mean(gray[top:bottom, left:right])
                if prev_light_intensity is not None:
                    light_diff = abs(light_intensity - prev_light_intensity)
                    no_light_change_frames = no_light_change_frames + 1 if light_diff < light_change_threshold else 0
                    if no_light_change_frames > max_no_light_change:
                        self.send_result("error", "No light change detected (static image).")
                        break
                prev_light_intensity = light_intensity

                if prev_face_location is not None:
                    dx = abs(prev_face_location[3] - left) + abs(prev_face_location[1] - right)
                    dy = abs(prev_face_location[0] - top) + abs(prev_face_location[2] - bottom)
                    no_movement_frames = no_movement_frames + 1 if dx < movement_threshold and dy < movement_threshold else 0
                    if no_movement_frames >= max_no_movement:
                        self.send_result("error", "No movement detected (photo).")
                        break
                prev_face_location = face_locations[0]

                rects = detector(gray, 0)
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    leftEye = shape[36:42]
                    rightEye = shape[42:48]
                    ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0

                    if ear < 0.22:
                        eye_closed_frames += 1
                    else:
                        if eye_closed_frames >= min_closed_frames:
                            blink_count += 1
                        eye_closed_frames = 0

                if blink_count >= required_blinks:
                    # Lưu hình ảnh khuôn mặt đã đăng ký
                    image_path = f"face_data/{self.emp_no}.png"
                    if not os.path.exists("face_data"):
                        os.makedirs("face_data")
                    cv2.imwrite(image_path, face_crop)

                    # Mã hóa khuôn mặt và lưu trữ
                    face_encoding = self.encode_face(rgb_frame, face_locations, face_crop)
                    if face_encoding:
                        folder_path = "face_encodings"
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                            # Xử lý cho Windows
                            if platform.system() == "Windows":
                                ctypes.windll.kernel32.SetFileAttributesW(folder_path, 0x02)

                        file_path = os.path.join(folder_path, f"{self.emp_no}.txt")
                        with open(file_path, "wb") as f:
                            f.write(face_encoding)

                        cap.release()
                        cv2.destroyAllWindows()
                        self.send_result("success", "Face registered successfully.", image_path, file_path)
                        return
                    else:
                        self.send_result("error", "Encoding failed.")
                        return

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            self.send_result("cancelled", "Process stopped by user.")
        except Exception as e:
            self.send_result("error", str(e))
