import cv2
import mediapipe as mp
import time
import json
import os

class CameraCollector:
    """Collect camera frames at fixed rate (~100Hz) with unified timestamps"""

    def __init__(self, save_dir="camera_data", target_fps=100):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.sample_count = 0
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps  # seconds per frame

    def collect(self, duration_seconds=60):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"\nStarting camera collection for {duration_seconds}s at ~{self.target_fps}Hz...")
        start_time = time.time()
        next_frame_time = start_time

        try:
            while time.time() - start_time < duration_seconds:
                current_time = time.time()
                # Sleep until the next frame time
                if current_time < next_frame_time:
                    time.sleep(next_frame_time - current_time)
                    continue

                timestamp_us = next_frame_time  # unified timestamp with BLE
                next_frame_time += self.frame_interval  # schedule next frame

                success, frame = cap.read()
                if not success:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_hands.process(frame_rgb)

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        score = handedness.classification[0].score

                        # Mirror left-hand to right-hand coordinates
                        landmarks = [[1.0 - lm.x, lm.y, lm.z] if label == "Left" else [lm.x, lm.y, lm.z]
                                     for lm in hand_landmarks.landmark]

                        sample = {
                            'timestamp': timestamp_us,
                            'landmarks': landmarks,
                            'hand': label,
                            'confidence': score
                        }

                        filename = f"{self.save_dir}/camera_{int(timestamp_us*1e6)}.json"
                        with open(filename, 'w') as f:
                            json.dump(sample, f)

                        self.sample_count += 1

                        # Draw landmarks on frame
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # Optional visualization
                cv2.putText(frame, f"Camera samples: {self.sample_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow('NEURA Camera Collection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
        print(f"Camera collection complete. Total samples: {self.sample_count}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--output", type=str, default="camera_data")
    parser.add_argument("--fps", type=int, default=100)
    args = parser.parse_args()

    collector = CameraCollector(save_dir=args.output, target_fps=args.fps)
    collector.collect(duration_seconds=args.duration)


if __name__ == "__main__":
    main()
