# import cv2
# import os
#
#
# DATASET_PATH = r"C:\Users\harshada\OneDrive\Desktop\Team chocos\Codes\VoilenceDetect\Real Life Violence Dataset"
# OUTPUT_PATH = r"C:\Users\harshada\OneDrive\Desktop\Team chocos\Codes\VoilenceDetect\ProcessFrames"
#
#
# os.makedirs(os.path.join(OUTPUT_PATH, "Violence"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_PATH, "NonViolence"), exist_ok=True)
#
#
# def extract_frames(video_path, label):
#     cap = cv2.VideoCapture(video_path)
#     count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_path = os.path.join(OUTPUT_PATH, label, f"frame_{os.path.basename(video_path)}_{count}.jpg")
#         cv2.imwrite(frame_path, frame)
#         count += 1
#     cap.release()
#
#
#
#
# for category in ["Violence", "NonViolence"]:
#     folder_path = os.path.join(DATASET_PATH, category)
#
#     if not os.path.exists(folder_path):
#         print(f" Error: Folder {folder_path} not found!")
#         continue
#
#         for video in os.listdir(folder_path):
#             video_path = os.path.join(folder_path, video)
#
#             if not os.path.isfile(video_path):
#                 print(f"Skipping non-file: {video_path}")
#                 continue
#
#             print(f"Processing: {video_path}")
#             extract_frames(video_path, category)
#             print(f" Extracted frames from {video}")
#
#     else:
#         print(" Frame extraction complete!")
