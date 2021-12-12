'''
Main Run Code
'''
import cv2
import os

from preprocess import preprocess
import local_common as cm
import model_tf2 as model

EPOCH_PATH = r'.\epochs'

def run():
    epoch_videos = [x for x in os.listdir(EPOCH_PATH) if x.endswith('.mkv')]
    
    for vidname in epoch_videos:
        vidpath = os.path.join(EPOCH_PATH, vidname)
        print(vidpath)

        frame_count = cm.frame_count(vidpath)

        cap = cv2.VideoCapture(vidpath)

        ret = True
        for frame in range(frame_count):
            # read image
            ret, img = cap.read()
            if not ret:
                break

            # preprocess image
            img = preprocess(img)

            # deg = model_tf1.y.eval(feed_dict={model_tf1.x: [img], model_tf1.keep_prob: 1.0})[0][0]
            m = model.build_model(img.shape)

            # show image
            cv2.imshow('Example - Show image in window',img)
            
            cv2.waitKey(10) # waits until a key is pressed
        break
    
    cv2.destroyAllWindows() # destroys the window showing image

if __name__ == '__main__':
    run()