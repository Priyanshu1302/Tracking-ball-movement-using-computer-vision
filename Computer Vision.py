import cv2
import numpy as np
import time

def process_video(input_path, output_video_path, output_txt_path):
    # Load the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Define quadrants
    quadrant_width = width // 2
    quadrant_height = height // 2

    quadrants = {
        1: (0, 0, quadrant_width, quadrant_height),
        2: (quadrant_width, 0, width, quadrant_height),
        3: (0, quadrant_height, quadrant_width, height),
        4: (quadrant_width, quadrant_height, width, height),
    }

    # Define color thresholds for ball detection
    color_ranges = {
        "red": ((0, 120, 70), (10, 255, 255)),
        "green": ((36, 25, 25), (70, 255, 255)),
        "blue": ((94, 80, 2), (126, 255, 255)),
    }

    # Store event logs
    event_log = []

    # Process video frames
    frame_count = 0
    ball_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / fps

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Filter by area to ignore noise
                area = cv2.contourArea(contour)
                if area < 500:
                    continue

                # Get bounding box and centroid
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2

                # Draw the ball and its color
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Check which quadrant
                for quadrant, (qx1, qy1, qx2, qy2) in quadrants.items():
                    if qx1 <= cx <= qx2 and qy1 <= cy <= qy2:
                        if color not in ball_positions or ball_positions[color] != quadrant:
                            event_type = "Entry" if color not in ball_positions else "Exit"
                            event_log.append(f"{timestamp:.2f}, {quadrant}, {color}, {event_type}")
                            ball_positions[color] = quadrant
                        break

        # Draw quadrant lines
        cv2.line(frame, (quadrant_width, 0), (quadrant_width, height), (255, 255, 255), 2)
        cv2.line(frame, (0, quadrant_height), (width, quadrant_height), (255, 255, 255), 2)

        # Write processed frame to output
        out.write(frame)

        # Display frame (optional)
        # cv2.imshow("Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Write event log to file
    with open(output_txt_path, 'w') as f:
        f.write("Time, Quadrant Number, Ball Colour, Type\n")
        f.write("\n".join(event_log))

    print("Processing complete.")
    print(f"Processed video saved to {output_video_path}")
    print(f"Event log saved to {output_txt_path}")

# Example usage
process_video(
    r"C:\Users\Priyanshu Malik\Downloads\AI Assignment video.mp4", 
    r"C:\Users\Priyanshu Malik\Downloads\processed_video.mp4", 
    r"C:\Users\Priyanshu Malik\Downloads\events.txt"
)
