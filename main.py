import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import cv2 as cv
from datetime import datetime
from collections import defaultdict
import time
import numpy as np

model = YOLO('best.pt')

video_path = "Traffic_Video.mp4" # Use actual filename
cap = cv.VideoCapture(video_path)

os.makedirs("violations", exist_ok=True)

object_y_hist = defaultdict(list)
saved_ids = set()

fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

start_time = time.time()
frame_count = 0
frame_skip = 5

# OPTIONAL: Save the video (Not required for live recording)
output_path = "Annotated_Video.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v') # codec
output_fps = fps / frame_skip
out = cv.VideoWriter(output_path, fourcc, output_fps, (854, 480))

# Function to check if the signal is red
# NOTE: This function is hardcoded according to the video, however external data of an actual traffic signal can be used instead.
def is_red_light():
    current_pos_ms = cap.get(cv.CAP_PROP_POS_MSEC)
    current_pos_seconds = current_pos_ms / 1000.0
    return current_pos_seconds > 12

# Function to draw the traffic light display
def draw_traffic_light(frame, is_red_light):
	# Draw box
	cv.rectangle(frame, (800,10), (850,110), (50,50,50), -1)
	cv.rectangle(frame, (800,10), (850,110), (255,255,255), 2)

	# Red light
	color = (0,0,255) if is_red_light else (0,0,50)
	cv.circle(frame, (825,35), 15, color, -1)

	# Green light
	color = (0, 255, 0) if not is_red_light else (0, 50, 0)
	cv.circle(frame, (825, 85), 15, color, -1)

violation_timers = {} # Track frames since violation of the vehicle
flash_duration = int(fps/frame_skip * 2) # Flash for 2 seconds

# Check if vehichle needs to be flashed
def flash_vehicle(vehicle_id):
	if vehicle_id in violation_timers:
		time_since_violation = violation_timers[vehicle_id]

		# Stop flashing after the mentioned duration
		if time_since_violation > flash_duration:
			return False

		# Flash after every two processed frames 
		flash_pattern = (time_since_violation // 2) % 2 == 0

		# Increment the counter for next call
		violation_timers[vehicle_id] += 1

		return flash_pattern

	else:
		return False

number_of_violations = 0

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	frame_count += 1
	if frame_count % frame_skip != 0:
		continue

	# Resize frames
	frame_resized = cv.resize(frame, (854, 480)) #(854, 480)

	# Track vehicles
	results = model.track(frame_resized, persist=True, classes=[0,1,2,3,4,5,7,8,9,10,11,12])

	# Plot model results
	annotated_frame = results[0].plot()
	#print(frame.shape)

	# Draw wait lines for the vehicles
	cv.line(annotated_frame, (10,300), (844,315), (0,0,255), thickness=2)
	cv.line(annotated_frame, (844,0), (844,315), (0,0,255), thickness=2)
	cv.line(annotated_frame, (10,0), (10,300), (0,0,255), thickness=2)


	# Check if obejects are crossing the line on red light
	if is_red_light():
		if results[0].boxes.id is not None:
			for box in results[0].boxes:
				id = int(box.id)
				x1, y1, x2, y2 = map(int, box.xyxy[0]) # Get xy values of the object(vehicle) from the frame
				#center_y = int((y1+y2)/2) # Find the y center of the vehicle
				start_y = y2 - 20 # Find the start of the box

				object_y_hist[id].append(start_y) # Store the x position history of the vehicle
				if len(object_y_hist[id]) >= 2:
					# Only store previous and current position
					prev_y = object_y_hist[id][-2]
					curr_y = object_y_hist[id][-1]

					# Line threshold (hardcoded, must be changed based on real world scenario)
					line_y = 310

					# Check if line crossed
					if prev_y < line_y and curr_y >= line_y and id not in saved_ids:
						number_of_violations += 1
						violation_timers[id] = 0
						cropped = frame_resized[y1-5:y2+5, x1-5:x2+5]
						timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
						filename = f"violations/vehicle_{id}_{timestamp}.jpg"
						# Save image of the vehicle
						cv.imwrite(filename, cropped)
						print(f"[SAVED] {filename}")
						saved_ids.add(id)

				# Display flash graphics
				if flash_vehicle(id):
					cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
					cv.putText(annotated_frame, "VIOLATION!", (x2-80, y2+25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
					
	# Display Graphics
	draw_traffic_light(annotated_frame, is_red_light())
	cv.rectangle(annotated_frame, (5, 4), (275, 45), (0,0,0), 2)
	cv.rectangle(annotated_frame, (5, 4), (275, 45), (255,255,255), -1)
	cv.putText(annotated_frame, f"Active Vehicle Count: {len(results[0].boxes) if results[0].boxes.id is not None else 0}", (25,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
	cv.putText(annotated_frame, f"Violations: {number_of_violations}", (25,40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)


	out.write(annotated_frame) # OPTIONAL: save
	cv.imshow("Vehicle Detection", annotated_frame) # Display the frame

	if cv.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
out.release()

cv.destroyAllWindows()
