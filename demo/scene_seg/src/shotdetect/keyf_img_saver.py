from __future__ import print_function

import logging
import math
import os
import pdb
from string import Template

import cv2

from shotdetect.platform import get_cv2_imwrite_params, tqdm


def get_output_file_path(file_path, output_dir=None):
    """ Get Output File Path: Gets full path to output file passed as argument, in
    the specified global output directory (scenedetect -o/--output) if set, creating
    any required directories along the way.

    Args:
        file_path (str): File name to get path for.  If file_path is an absolute
            path (e.g. starts at a drive/root), no modification of the path
            is performed, only ensuring that all output directories are created.
        output_dir (Optional[str]): An optional output directory to override the
            global output directory option, if set.

    Returns:
        (str) Full path to output file suitable for writing.

    """
    output_directory = None
    if file_path is None:
        return None
    output_dir = output_directory if output_dir is None else output_dir
    # If an output directory is defined and the file path is a relative path, open
    # the file handle in the output directory instead of the working directory.
    if output_dir is not None and not os.path.isabs(file_path):
        file_path = os.path.join(output_dir, file_path)
    # Now that file_path is an absolute path, let's make sure all the directories
    # exist for us to start writing files there.
    try:
        os.makedirs(os.path.split(os.path.abspath(file_path))[0])
    except OSError:
        pass
    return file_path


def extract_shots_chunk_worker(video_paths, shot_chunk, output_dir, image_name_template, shot_num_format, image_num_format, imwrite_param, num_images):
    import cv2
    from shotdetect.video_manager import VideoManager
    from shotdetect.keyf_img_saver import get_output_file_path
    from string import Template

    filename_template = Template(image_name_template)
    
    # Initialize local video manager in worker process
    video_manager = VideoManager(video_paths)
    video_manager.set_downscale_factor(1)
    video_manager.start()
    
    middle_images = num_images - 2
    for shot_idx, (start_time, end_time) in shot_chunk:
        timecodes = []
        if num_images == 1:
            duration = end_time - start_time
            timecodes.append(start_time + int(duration.get_frames() / 2))
        else:
            timecodes.append(start_time)
            if middle_images > 0:
                duration = (end_time.get_frames() - 1) - start_time.get_frames()
                duration_increment = int(duration / (middle_images + 1))
                for j in range(middle_images):
                    timecodes.append(start_time + ((j+1) * duration_increment))
            timecodes.append(end_time - 1)
        
        for j, image_timecode in enumerate(timecodes):
            video_manager.seek(image_timecode)
            video_manager.grab()
            ret_val, frame_im = video_manager.retrieve()
            if ret_val:
                out_path = get_output_file_path(
                    '%s.jpg' % (filename_template.safe_substitute(
                        SHOT_NUMBER=shot_num_format % (shot_idx),
                        IMAGE_NUMBER=image_num_format % (j)
                    )),
                    output_dir=output_dir
                )
                cv2.imwrite(out_path, frame_im, imwrite_param)
                
    video_manager.release()
    return True


def generate_images(video_manager, shot_list, output_dir, num_images=3,
                    image_name_template='shot_${SHOT_NUMBER}_img_${IMAGE_NUMBER}',
                    ):
    '''
        Args:
            num_images: number of keyframes
    '''
    assert num_images >= 1
    os.makedirs(output_dir, exist_ok=True)
    if num_images == 1:
        image_name_template = 'shot_${SHOT_NUMBER}'
    else:
        pass
    if not shot_list:
        return

    quiet_mode = False
    imwrite_params = get_cv2_imwrite_params()
    image_param = None
    image_extension = 'jpg'
    imwrite_param = []
    if image_param is not None:
        imwrite_param = [imwrite_params[image_extension], image_param]

    # Save video_paths and release main video_manager handles
    video_paths = video_manager.get_video_paths()
    video_manager.release()

    logging.info('Generating output images parallelly (%d per shot)...', num_images)

    shot_num_format = '%0'
    shot_num_format += str(max(4, math.floor(math.log(len(shot_list), 10)) + 1)) + 'd'
    image_num_format = '%0'
    image_num_format += str(math.floor(math.log(num_images, 10)) + 1) + 'd'

    # Split shot_list into chunks for parallel processes
    num_workers = min(16, len(shot_list))
    shot_list_indexed = list(enumerate(shot_list))
    chunks = [shot_list_indexed[i::num_workers] for i in range(num_workers)]

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk in chunks:
            if not chunk:
                continue
            futures.append(executor.submit(
                extract_shots_chunk_worker,
                video_paths, chunk, output_dir, image_name_template,
                shot_num_format, image_num_format, imwrite_param, num_images
            ))
        
        # Wait for all futures to complete
        for future in futures:
            future.result()

    # Re-initialize main video_manager
    video_manager.reset()


def generate_images_txt(shot_list, output_dir, num_images=5):
    assert num_images >= 3
    timecode_list = dict()
    for i in range(len(shot_list)):
        timecode_list[i] = []
    middle_images = num_images - 2
    for i, (start_time, end_time) in enumerate(shot_list):
        timecode_list[i].append(start_time)
        if middle_images > 0:
            duration = (end_time.get_frames() - 1) - start_time.get_frames()
            duration_increment = None
            duration_increment = int(duration / (middle_images + 1))  # middle_images + 1 is the middle segment number
            for j in range(middle_images):
                timecode_list[i].append(start_time + ((j+1) * duration_increment))

        # End FrameTimecode is always the same frame as the next shot's start_time
        # (one frame past the end), so we need to subtract 1 here.
        timecode_list[i].append(end_time - 1)

    frames_list = []
    for i in timecode_list:
        frame_list = []
        for j, image_timecode in enumerate(timecode_list[i]):
            frame_list.append(image_timecode.get_frames())
        frames_item = "{} {} ".format(frame_list[0], frame_list[-1])
        for i in range(num_images-2):
            frames_item += "{} ".format(frame_list[i+1])
        frames_list.append(frames_item[:-1])

    with open(output_dir, 'w') as f:
        for frames in frames_list:
            f.write("{}\n".format(frames))
