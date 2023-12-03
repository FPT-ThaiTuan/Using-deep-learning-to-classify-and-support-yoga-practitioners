import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Create a pose instance
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
     # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360
    return angle

def correct_feedback(model,video='0'):
    # Load video
    cap = cv2.VideoCapture(video)  # Replace with your video path

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    # Your accurate angle list
    accurate_angle_list = [101, 95, 187, 178, 152, 128, 116, 186]
    correction_value = 13

    angle_name_list = ["R-shoulder", "L-shoulder", "R-elbow", "L-elbow", "R-hip", "L-hip", "R-knee", "L-knee"]
    angle_coordinates = [[24, 12, 14], [13, 11, 23], [16, 14, 12], [11, 13, 15], [23, 24, 26], [24, 23, 25], [24, 26, 28], [23, 25, 27]]
    correction_value = 10

    fps_time = 0
    global flat
    flat = [0.0 for i in range(36)]
    while cap.isOpened():
        ret_val, image = cap.read()

        if not ret_val:
            break

        # Convert the image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get the pose landmarks
        results = pose.process(image_rgb)

        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
                
            correct_angle_count = 0
            for itr in range(8):
                point_a = (int(landmarks[angle_coordinates[itr][0]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][0]].y * image.shape[0]))
                point_b = (int(landmarks[angle_coordinates[itr][1]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][1]].y * image.shape[0]))
                point_c = (int(landmarks[angle_coordinates[itr][2]].x * image.shape[1]),
                        int(landmarks[angle_coordinates[itr][2]].y * image.shape[0]))

                angle_obtained = calculate_angle(point_a, point_b, point_c)

                if angle_obtained < accurate_angle_list[itr] - correction_value:
                    status = "more"
                elif accurate_angle_list[itr] + correction_value < angle_obtained:
                    status = "less"
                else:
                    status = "OK"
                    correct_angle_count += 1

                # cv2.putText(image, f"{angle_name_list[itr]}: {status}", (pos_on_screen[itr % 2][0], (itr // 2 + 1) * 70),
                #             cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                # Display status
                status_position = (point_b[0] - int(image.shape[1] * 0.03), point_b[1] + int(image.shape[0] * 0.03))
                cv2.putText(image, f"{status}", status_position, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

                # Draw lines connecting the key points
                cv2.line(image, point_a, point_b, (0, 255, 0), 3)
                cv2.line(image, point_b, point_c, (0, 255, 0), 3)

                # Display angle values on the image
                cv2.putText(image, f"{angle_name_list[itr]}: {angle_obtained:.2f} degrees",
                            (point_b[0] - 50, point_b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # Draw the entire pose on the person
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Classification based on the pose landmarks
            temp = []
            for j in landmarks:
                temp = temp + [j.x, j.y, j.z, j.visibility]
            y = model.predict([temp])
            name = str(y[0])
            
            # Display the classification result in the bottom-left corner
            (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.rectangle(image, (10, image.shape[0] - 30), (10 + w, image.shape[0] - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, name, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
 

            # Calculate text positions based on the frame size
            posture_position = (int(image.shape[1] * 0.4), int(image.shape[0] * 0.05))
            fps_position = (int(image.shape[1] * 0.4), int(image.shape[0] * 0.1))

            posture = "CORRECT" if correct_angle_count > 6 else "WRONG"
            cv2.putText(image, f"Posture: {posture}", posture_position, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            cv2.putText(image, f"FPS: {1.0 / (time.time() - fps_time)}", fps_position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            # posture = "CORRECT" if correct_angle_count > 6 else "WRONG"
            # cv2.putText(image, f"Posture: {posture}", (590, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            # cv2.putText(image, f"FPS: {1.0 / (time.time() - fps_time)}", (590, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 255, 0), 2)
            cv2.imshow('Mediapipe Pose Estimation', image)
            fps_time = time.time()

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
