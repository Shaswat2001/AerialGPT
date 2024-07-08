class UserVision:
    def __init__(self, vision):
        self.vision = vision
        self.image = None

    def save_pictures(self, args):
        #print("saving picture")
        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            self.image = img