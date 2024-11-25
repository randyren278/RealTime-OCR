import os
import sys
from threading import Thread
from datetime import datetime

import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def capture_image(frame, captures=0):
    """
    Capture a .jpg during CV2 video stream. Saves to a folder /images in the working directory.
    """
    cwd_path = os.getcwd()
    images_dir = os.path.join(cwd_path, 'images')
    os.makedirs(images_dir, exist_ok=True)

    now = datetime.now()
    name = f"OCR_Capture_{now.strftime('%Y-%m-%d_%H-%M-%S')}-{captures + 1}.jpg"
    path = os.path.join(images_dir, name)
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
        Thread(target=self.update, args=(), daemon=True).start()
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

class TrOCRProcessorClass:
    def __init__(self):
        self.stopped = False
        self.exchange = None
        self.result = None

        # Load the largest TrOCR model for printed text
        self.model_name = "microsoft/trocr-large-printed"
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU for inference")
            self.model = self.model.to(self.device)
            self.fp16 = True  # Enable half-precision
        else:
            self.device = torch.device("cpu")
            print("Using CPU for inference")
            self.fp16 = False

        if self.fp16:
            self.model = self.model.half()

    def start(self):
        Thread(target=self.process_frames, args=(), daemon=True).start()
        return self

    def set_exchange(self, video_stream):
        self.exchange = video_stream

    def process_frames(self):
        while not self.stopped:
            if self.exchange is not None and self.exchange.frame is not None:
                frame = self.exchange.frame.copy()

                # Convert frame to PIL image
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Optionally resize the image for faster processing
                # image = image.resize((800, 600))

                # Prepare image input
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                if self.fp16:
                    pixel_values = pixel_values.half()

                # Generate text
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                self.result = generated_text
            else:
                torch.cuda.empty_cache()
                continue

    def stop(self):
        self.stopped = True

def trocr_stream(source: int = 0):
    """
    Starts the video stream and OCR processing using TrOCR.
    """
    captures = 0

    video_stream = VideoStream(source).start()
    img_wi, img_hi = video_stream.get_video_dimensions()

    trocr_processor = TrOCRProcessorClass().start()
    trocr_processor.set_exchange(video_stream)

    # Main display loop
    print("\nPress 'c' to capture an image. Press 'q' to quit the video stream.\n")
    while True:
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            video_stream.stop()
            trocr_processor.stop()
            print("OCR stream stopped\n")
            print(f"{captures} image(s) captured and saved to the current directory")
            break

        if video_stream.frame is None:
            continue

        frame = video_stream.frame.copy()

        # Display the recognized text at the top of the frame
        if trocr_processor.result is not None:
            text = trocr_processor.result.strip()
            # Limit the text length if necessary
            max_chars = 80
            if len(text) > max_chars:
                text = text[:max_chars] + '...'

            # Prepare text for display
            y0, dy = 30, 30
            lines = text.split('\n')
            for i, line in enumerate(lines):
                y = y0 + i * dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Photo capture
        if pressed_key == ord('c'):
            captures = capture_image(frame, captures)

        cv2.imshow("TrOCR Text Recognition", frame)
