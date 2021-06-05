import cv2

class VideoCaptureWrapper():
    
    def __init__(self, video_path, start=0, end=np.inf, step=1):
        self.video = cv2.VideoCapture(video_path)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start)
        self.step = step
        self.end = end
        self.i = start
        
    def read(self):
        if self.i <= self.end:
            for _ in range(self.step):
                self.video.read()
                self.i += 1
            return self.video.read()
        return False, None