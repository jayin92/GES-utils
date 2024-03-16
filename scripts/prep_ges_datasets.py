import os
import argparse
import cv2

# Extracts the frames from the video, and remove the watermark by combine two videos
def extract_frames(video_path_left, video_path_right, frames_path, lowres=False):
    os.makedirs(frames_path, exist_ok=True)
    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)
    count = 0
    while cap_left.isOpened() and cap_right.isOpened():
        ret1, frame1 = cap_left.read()
        ret2, frame2 = cap_right.read()
        
        if ret1 and ret2:
            print(f"Extracting frame {count}")
            res = frame1
            res[1877:2160-1, 0:1500] = frame2[1877:2160-1, 0:1500]
            if lowres:
                res = cv2.resize(res, (int(res.shape[1]/2), int(res.shape[0]/2)))
        
            cv2.imwrite(os.path.join(frames_path, f"frame{count}.jpg"), res)
            count += 1
            # cv2.imwrite(frames_path + "frame%d.jpg" % count, frame2)
            # count += 1
        else:
            break
    cap_left.release()
    cap_right.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", help="path to the left video", required=True)
    parser.add_argument("--right", help="path to the right video", required=True)
    parser.add_argument("--output", help="path to the frames", required=True)
    parser.add_argument("--lowres", action="store_true", help="low resolution video", required=False, default=False)

    args = parser.parse_args()
    extract_frames(args.left, args.right, args.output, args.lowres)