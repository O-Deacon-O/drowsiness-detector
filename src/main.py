import config
import cv2 as cv
import face_mesh_module as fmm
import time
import utils

def main() -> None:
    face_mesh_detector = None
    video_capture = cv.VideoCapture(0)

    try:
        if not video_capture.isOpened():
            print("Could not open video capture.")
            return

        # === Configure Camera Resolution ===
        video_capture.set(cv.CAP_PROP_FRAME_WIDTH, config.TARGET_WIDTH)
        video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, config.TARGET_HEIGHT)

        face_mesh_detector = fmm.FaceMeshDetector(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        previous_time = time.monotonic()
        closed_start_time = None

        # === Main Video Loop ===
        while True:
            success, frame = video_capture.read()

            if not success:
                print("End of video stream.")
                break

            # === Process Frame for Face Mesh Detection ===
            frame.flags.writeable = False  # Lock the frame memory to improve performance
            frame, faces = face_mesh_detector.find_face_mesh(frame)
            frame.flags.writeable = True

            # === Drowsiness Detection Logic ===
            if faces:
                face = faces[0]

                left_eye_ratio = face_mesh_detector.get_eye_ratio(face, config.LEFT_EYE)
                right_eye_ratio = face_mesh_detector.get_eye_ratio(face, config.RIGHT_EYE)

                if left_eye_ratio is None or right_eye_ratio is None:
                    closed_start_time = None
                    continue
                
                avg_ratio = (left_eye_ratio + right_eye_ratio) / 2

                # === Draw Eye Landmarks and Connecting Lines ===
                for (i1, i2) in [(159, 145), (386, 374)]:
                    x1, y1 = face[i1]
                    cv.circle(frame, (x1, y1), 2, (0, 0, 255), cv.FILLED)

                    x2, y2 = face[i2]
                    cv.circle(frame, (x2, y2), 2, (0, 0, 255), cv.FILLED)

                    cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # === Drowsiness Detection ===
                if avg_ratio < config.CLOSED_THRESHOLD:
                    if closed_start_time is None:
                        closed_start_time = time.monotonic()

                    elapsed = time.monotonic() - closed_start_time

                    cv.putText(frame, f"CLOSED: {elapsed:.2f}s", (10, 70), cv.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALES[0], config.TEXT_COLORS[2], config.TEXT_THICKNESSES[0])

                    if elapsed >= config.CLOSED_TIME_REQUIRED:
                        cv.putText(frame, "WAKE UP!!!", (100, 150), cv.FONT_HERSHEY_COMPLEX, config.TEXT_SCALES[2], config.TEXT_COLORS[2], config.TEXT_THICKNESSES[1])
                else:
                    closed_start_time = None

                cv.putText(frame, f"EYE RATIO: {avg_ratio:.4f}", (10, 110), cv.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALES[0], config.TEXT_COLORS[0], config.TEXT_THICKNESSES[0])
            else:
                closed_start_time = None

            # === Calculate and Display FPS ===
            fps, previous_time = utils.calculate_fps(previous_time)
            cv.putText(frame, f"FPS: {int(fps)}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALES[1], config.TEXT_COLORS[1], config.TEXT_THICKNESSES[0])

            # === Display Frame ===
            cv.imshow("Drowsiness Detector", frame)

            # === Check for Quit Command 'Q' ===
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # === Cleanup ===
        if face_mesh_detector is not None:
            face_mesh_detector.close()

        video_capture.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()