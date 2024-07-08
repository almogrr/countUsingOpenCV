import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):
    """
    Count the number of fingers extended on a hand.
    
    Args:
        hand_landmarks: The landmarks of the hand detected by MediaPipe.

    Returns:
        Integer representing the number of fingers that are open.
    """
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb.
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers.
    for finger_id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[finger_id]].y < hand_landmarks.landmark[tip_ids[finger_id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)  # Return the number of fingers that are open.

def main():
    """
    Main function to run the hand and finger detection application.
    """
    # Initialize the webcam.
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display.
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands.
        result = hands.process(rgb_frame)

        # Initialize counters for each hand.
        hand_count = 0
        hand_finger_counts = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks.
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Count fingers.
                total_fingers = count_fingers(hand_landmarks)
                hand_finger_counts.append(total_fingers)
                hand_count += 1

                # Display the finger count for each hand.
                hand_label = f'Hand {hand_count}: {total_fingers} Fingers'
                cv2.putText(frame, hand_label, (10, 70 + (hand_count - 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame.
        cv2.imshow('Hand Tracking', frame)

        # Press 'q' to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
