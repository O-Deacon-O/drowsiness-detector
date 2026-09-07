import cv2 as cv
import mediapipe as mp

from numpy.typing import NDArray

class FaceMeshDetector():
    """Detect facial landmarks using MediaPipe's Face Mesh solution."""
    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        """Initialize the MediaPipe Face Mesh detector."""
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def find_face_mesh(self, frame: NDArray, draw: bool = False) -> tuple[NDArray, list[list[tuple[int, int]]]]:
        """Detect faces and return landmark coordinates."""
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        faces = []

        if results.multi_face_landmarks:
            h, w, _ = frame.shape

            for face_landmarks in results.multi_face_landmarks:
                face = []

                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    face.append((x, y))

                faces.append(face)

                if draw:
                    self.mp_draw.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION, self.draw_spec, self.draw_spec)

        return frame, faces

    def get_eye_ratio(self, face: list[tuple[int, int]], indices: tuple[int, int, int, int]) -> float | None:
        """Calculate the eye aspect ratio."""
        v1, v2, h1, h2 = indices

        x1, y1 = face[v1]
        x2, y2 = face[v2]
        x3, y3 = face[h1]
        x4, y4 = face[h2]

        
        dv = (x2 - x1) ** 2 + (y2 - y1) ** 2 # Vertical distance squared
        dh = (x4 - x3) ** 2 + (y4 - y3) ** 2 # Horizontal distance squared

        if dh == 0:
            return None

        return dv / dh 

    def close(self) -> None:
        """Release resources used by the FaceMeshDetector."""
        self.face_mesh.close()