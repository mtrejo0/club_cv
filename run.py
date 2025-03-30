from ultralytics import YOLO
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def enhance_method1(frame):
    """CLAHE with high contrast"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    # High contrast adjustment
    alpha = 1.5  # Higher contrast
    beta = 15    # Higher brightness
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    return enhanced

def enhance_method2(frame):
    """Gamma correction with sharpening"""
    # Gamma correction
    gamma = 1.5
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
    enhanced = cv2.LUT(frame, lookUpTable)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced

def enhance_method3(frame):
    """Histogram equalization with denoising"""
    # Convert to YUV
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Apply histogram equalization to Y channel
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    # Denoise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    return enhanced

# Initialize YOLO model
model = YOLO('yolov8x.pt')  # using the largest variant for better accuracy

# Open the video file
video = cv2.VideoCapture('video.mp4')
fps = video.get(cv2.CAP_PROP_FPS)

# Lists to store results for each method
results_method1 = {'timestamps': [], 'counts': [], 'name': 'CLAHE + High Contrast'}
results_method2 = {'timestamps': [], 'counts': [], 'name': 'Gamma + Sharpening'}
results_method3 = {'timestamps': [], 'counts': [], 'name': 'Histogram + Denoising'}

frame_count = 0
frames_to_skip = int(fps)  # process one frame per second

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frames_to_skip != 0:
        continue
        
    timestamp = frame_count / fps
    
    # Process with each enhancement method
    enhanced1 = enhance_method1(frame.copy())
    enhanced2 = enhance_method2(frame.copy())
    enhanced3 = enhance_method3(frame.copy())
    
    # Detect people in each enhanced frame
    results1 = model(enhanced1, classes=[0], conf=0.15)
    results2 = model(enhanced2, classes=[0], conf=0.15)
    results3 = model(enhanced3, classes=[0], conf=0.15)
    
    # Count detections and store results
    count1 = len(results1[0].boxes)
    count2 = len(results2[0].boxes)
    count3 = len(results3[0].boxes)
    
    results_method1['timestamps'].append(timestamp)
    results_method1['counts'].append(count1)
    results_method2['timestamps'].append(timestamp)
    results_method2['counts'].append(count2)
    results_method3['timestamps'].append(timestamp)
    results_method3['counts'].append(count3)
    
    # Display all versions
    cv2.imshow('Original', frame)
    cv2.imshow('Method 1: CLAHE + High Contrast', enhanced1)
    cv2.imshow('Method 2: Gamma + Sharpening', enhanced2)
    cv2.imshow('Method 3: Histogram + Denoising', enhanced3)
    
    # Draw boxes on the best performing method
    best_count = max(count1, count2, count3)
    if best_count == count1:
        best_frame = enhanced1
        best_results = results1
    elif best_count == count2:
        best_frame = enhanced2
        best_results = results2
    else:
        best_frame = enhanced3
        best_results = results3
        
    # Draw boxes on best frame
    for box in best_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
        confidence = float(box.conf[0])
        cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(best_frame, f'{confidence:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Best Detection', best_frame)
    
    print(f"Timestamp: {timestamp:.2f}s - Counts: Method1={count1}, Method2={count2}, Method3={count3}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Plot comparison graph
plt.figure(figsize=(15, 8))
plt.plot(results_method1['timestamps'], results_method1['counts'], 'b-', label=results_method1['name'])
plt.plot(results_method2['timestamps'], results_method2['counts'], 'r-', label=results_method2['name'])
plt.plot(results_method3['timestamps'], results_method3['counts'], 'g-', label=results_method3['name'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Time (seconds)')
plt.ylabel('Number of People Detected')
plt.title('People Count Comparison of Different Enhancement Methods')
plt.legend()
plt.tight_layout()
plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
