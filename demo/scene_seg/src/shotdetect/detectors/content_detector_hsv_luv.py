import pdb

import cv2
import numpy as np

from shotdetect.shot_detector import shotDetector


class ContentDetectorHSVLUV(shotDetector):
    """Detects fast cuts using changes in colour and intensity between frames.

    Since the difference between frames is used, unlike the ThresholdDetector,
    only fast cuts are detected with this method.  To detect slow fades between
    content shots still using HSV information, use the DissolveDetector.
    """

    def __init__(self, threshold=30.0, min_shot_len=15):
        super(ContentDetectorHSVLUV, self).__init__()
        self.hsv_threshold = threshold
        self.delta_hsv_gap_threshold = 10
        self.luv_threshold = 20
        self.hsv_weight = 5
        self.min_shot_len = min_shot_len  # minimum length of any given shot, in frames
        self.last_frame = None
        self.last_shot_cut = None
        self.last_hsv = None
        self._metric_keys = ['hsv_content_val', 'delta_hsv_hue', 'delta_hsv_sat', 'delta_hsv_lum','luv_content_val', 'delta_luv_hue', 'delta_luv_sat', 'delta_luv_lum']
        self.cli_name = 'detect-content'
        self.last_luv = None

    def process_frame(self, frame_num, frame_img):
        # type: (int, np.ndarray) -> List[int]
        """ Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        of single-frame RGB/grayscale intensity (thus cannot detect slow fades with this method).

        Args:
            frame_num (int): Frame number of frame that is being passed.

            frame_img (Optional[int]): Decoded frame image (np.ndarray) to perform shot
                detection on. Can be None *only* if the self.is_processing_required() method
                (inhereted from the base shotDetector class) returns True.

        Returns:
            List[int]: List of frames where shot cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """
        cut_list = []
        metric_keys = self._metric_keys
        _unused = ''

        if self.last_frame is not None:
            # Change in average of HSV (hsv), (h)ue only, (s)aturation only, (l)uminance only.
            delta_hsv_avg, delta_hsv_h, delta_hsv_s, delta_hsv_v = 0.0, 0.0, 0.0, 0.0
            delta_luv_avg, delta_luv_h, delta_luv_s, delta_luv_v = 0.0, 0.0, 0.0, 0.0

            if (self.stats_manager is not None and
                    self.stats_manager.metrics_exist(frame_num, metric_keys)):
                delta_hsv_avg, delta_hsv_h, delta_hsv_s, delta_hsv_v, delta_luv_avg, delta_luv_h, delta_luv_s, delta_luv_v = self.stats_manager.get_metrics(
                    frame_num, metric_keys)

            else:
                curr_hsv_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV)
                curr_luv_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2Luv)
                
                last_hsv_img = self.last_hsv
                last_luv_img = self.last_luv
                if last_hsv_img is None:
                    last_hsv_img = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2HSV)
                    last_luv_img = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2Luv)

                # Calculate HSV difference
                diff_hsv = cv2.absdiff(curr_hsv_img, last_hsv_img)
                mean_hsv = cv2.mean(diff_hsv)
                delta_hsv_h, delta_hsv_s, delta_hsv_v = mean_hsv[0:3]
                delta_hsv_avg = sum(mean_hsv[0:3]) / 3.0

                # Calculate LUV difference
                diff_luv = cv2.absdiff(curr_luv_img, last_luv_img)
                mean_luv = cv2.mean(diff_luv)
                delta_luv_h, delta_luv_s, delta_luv_v = mean_luv[0:3]
                delta_luv_avg = sum(mean_luv[0:3]) / 3.0

                self.last_hsv = curr_hsv_img
                self.last_luv = curr_luv_img

                if self.stats_manager is not None:
                    self.stats_manager.set_metrics(frame_num, {
                        metric_keys[0]: delta_hsv_avg,
                        metric_keys[1]: delta_hsv_h,
                        metric_keys[2]: delta_hsv_s,
                        metric_keys[3]: delta_hsv_v,
                        metric_keys[0+4]: delta_luv_avg,
                        metric_keys[1+4]: delta_luv_h,
                        metric_keys[2+4]: delta_luv_s,
                        metric_keys[3+4]: delta_luv_v,
                        })

                self.last_hsv = curr_hsv_img
                self.last_luv = curr_luv_img
            # pdb.set_trace()
            if delta_hsv_avg >= self.hsv_threshold and delta_hsv_avg - self.hsv_threshold >= self.delta_hsv_gap_threshold:
                # print(frame_num,delta_hsv_avg,delta_luv_avg)
                if self.last_shot_cut is None or (
                        (frame_num - self.last_shot_cut) >= self.min_shot_len):
                    cut_list.append(frame_num)
                    self.last_shot_cut = frame_num
            elif delta_hsv_avg >= self.hsv_threshold and delta_hsv_avg - self.hsv_threshold < self.delta_hsv_gap_threshold \
                and delta_luv_avg + self.hsv_weight * (delta_hsv_avg - self.hsv_threshold) > self.luv_threshold:
                # print(frame_num,delta_hsv_avg,delta_luv_avg)
                if self.last_shot_cut is None or (
                        (frame_num - self.last_shot_cut) >= self.min_shot_len):
                    cut_list.append(frame_num)
                    self.last_shot_cut = frame_num    

            if self.last_frame is not None and self.last_frame is not _unused:
                del self.last_frame

        # If we have the next frame computed, don't copy the current frame
        # into last_frame since we won't use it on the next call anyways.
        if (self.stats_manager is not None and
                self.stats_manager.metrics_exist(frame_num+1, metric_keys)):
            self.last_frame = _unused
        else:
            self.last_frame = frame_img.copy()
        # if len(cut_list) > 0:
        #     print(frame_num,cut_list)
        return cut_list
