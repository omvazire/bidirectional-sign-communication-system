"""
Threaded Webcam Stream for Asynchronous Frame Capture.
Prevents I/O blocking from dropping the overall FPS of the pipeline.
"""

import cv2
import threading
import time

class ThreadedCamera:
    """
    Continually reads frames from the webcam in a dedicated background thread.
    This ensures that MediaPipe and model inference never have to wait for the
    webcam's hardware I/O latency.
    """
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.stream.isOpened():
            raise ValueError(f"Unable to open camera source {src}")
            
        (self.grabbed, self.frame) = self.stream.read()
        
        self.stopped = False
        self.thread = None

    def start(self):
        """Start the background thread to read frames."""
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        """Keep looping and grabbing frames until stopped."""
        while True:
            if self.stopped:
                return
                
            (self.grabbed, self.frame) = self.stream.read()
            # If we need a tiny sleep to prevent 100% CPU lock up on this core:
            time.sleep(0.005) 

    def read(self):
        """Return the most recently read frame."""
        return self.grabbed, self.frame

    def release(self):
        """Stop the thread and release the camera stream."""
        self.stopped = True
        if self.thread:
            self.thread.join()
        self.stream.release()
