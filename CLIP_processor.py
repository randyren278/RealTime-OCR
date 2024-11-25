import os
from pathlib import Path
import sys
from threading import Thread
import time
from datetime import datetime

import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor as CLIP_Processor_Class, CLIPModel

def capture_image(frame, captures=0):
    """
    Capture a .jpg during CV2 video stream. Saves to a folder /images in the working directory.
    """
    cwd_path = os.getcwd()
    Path(cwd_path + '/images').mkdir(parents=False, exist_ok=True)

    now = datetime.now()
    name = "CLIP Capture " + now.strftime("%Y-%m-%d_%H-%M-%S") + '-' + str(captures + 1) + '.jpg'
    path = os.path.join('images', name)
    cv2.imwrite(path, frame)
    captures += 1
    print(f"Captured image: {name}")
    return captures

class VideoStream:
    """
    Class for grabbing frames from CV2 video capture.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print("Error: Could not open video source.")
            sys.exit()
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        """
        Starts the video stream in a separate thread.
        """
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """
        Continuously updates frames from the video source.
        """
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                self.stop()
                break

    def get_video_dimensions(self):
        """
        Gets the width and height of the video stream frames.
        """
        width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(width), int(height)

    def stop(self):
        """
        Stops the video stream.
        """
        self.stopped = True
        self.stream.release()

class CLIPProcessor:
    def __init__(self):
        self.stopped = False
        self.exchange = None
        self.result = None

        # Load the ImageNet class labels
        with open("imagenet_classes.txt", "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Prepend "a photo of a" to each label
        self.labels = [f"a photo of a {label}" for label in self.labels]

        # Load the CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIP_Processor_Class.from_pretrained("openai/clip-vit-large-patch14")

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU for inference")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for inference")
        self.model = self.model.to(self.device)

    def start(self):
        Thread(target=self.process_frames, args=()).start()
        return self

    def set_exchange(self, video_stream):
        self.exchange = video_stream

    def process_frames(self):
        frame_count = 0
        process_every_n_frames = 1  # Adjust this number as needed
        while not self.stopped:
            if self.exchange is not None and self.exchange.frame is not None:
                frame_count += 1
                if frame_count % process_every_n_frames == 0:
                    frame = self.exchange.frame.copy()

                    # Convert frame to PIL image
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Prepare inputs
                    inputs = self.processor(
                        text=self.labels, images=image, return_tensors="pt", padding=True
                    ).to(self.device)

                    # Perform inference
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits_per_image = outputs.logits_per_image  # (1, len(self.labels))
                        probs = logits_per_image.softmax(dim=1)  # (1, len(self.labels))

                    # Get the top prediction
                    top_prob, top_idx = probs[0].topk(1)
                    label = self.labels[top_idx.item()]
                    self.result = (label, top_prob.item())
                else:
                    time.sleep(0.01)  # Yield time to other threads

    def stop(self):
        self.stopped = True

def clip_stream(source: int = 0):
    """
    Starts the video stream and CLIP processing.
    """
    captures = 0

    video_stream = VideoStream(source).start()
    img_wi, img_hi = video_stream.get_video_dimensions()

    clip_processor = CLIPProcessor().start()
    clip_processor.set_exchange(video_stream)

    # Main display loop
    print("\nPress 'c' to capture an image. Press 'q' to quit the video stream.\n")
    while True:
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            video_stream.stop()
            clip_processor.stop()
            print("CLIP stream stopped\n")
            print(f"{captures} image(s) captured and saved to the current directory")
            break

        frame = video_stream.frame.copy()

        # Display the detection results on the frame
        if clip_processor.result is not None:
            label, prob = clip_processor.result
            text = f"{label[13:]}: {prob:.2f}"  # Remove "a photo of a " prefix
            cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Processing...", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Photo capture
        if pressed_key == ord('c'):
            captures = capture_image(frame, captures)

        cv2.imshow("CLIP Object Detection", frame)
