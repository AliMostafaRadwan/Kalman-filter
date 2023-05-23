import cv2
import numpy as np

def kalman_filter_video(input_video, output_video):
    # Create video capture object
    cap = cv2.VideoCapture(input_video)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Initialize Kalman filter
    kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Kalman filter
        prediction = kalman.predict()
        measurement = np.array([[np.float32(prediction[0])],
                                [np.float32(prediction[1])]])
        kalman.correct(measurement)
        estimation = kalman.predict()

        # Draw estimated position on the frame
        cv2.circle(frame, (int(estimation[0]), int(estimation[1])), 5, (0, 255, 0), -1)

        # Write frame to output video
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('Kalman Filter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_video = 'input_video.mp4'
output_video = 'filtered_video.avi'
kalman_filter_video(input_video, output_video)
