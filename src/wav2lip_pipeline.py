"""
Given a list of audio paths, generate a list of video paths using Wav2Lip.
"""

from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
from src import face_detection
from src.wav2lip import Wav2Lip, load_model
import platform
import librosa
from src.audio_utils import melspectrogram
from src.utils.video import images2video, add_audio_to_video



def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def face_detect(images, args):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen(frames, mels, args):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames, args) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]], args)
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))



def process_audio_file(args, audio_path, output_dir):
    """Process a single audio file and save the result to the output directory."""
    # Create output filename based on input audio filename
    audio_filename = os.path.basename(audio_path)
    audio_dir = args.audio_dir
    relative_audio_path = os.path.relpath(audio_path, audio_dir)
    output_filename = os.path.splitext(relative_audio_path)[0] + '_wav2lip.mp4'
    output_path = os.path.join(output_dir, output_filename)
    
    # Update args with current audio file and output path
    args.audio = audio_path
    args.outfile = output_path
    
    # Process the audio file
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
        if args.resize_factor > 1:
            full_frames = [cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor)) for frame in full_frames]

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)
        
    print ("Number of frames available for inference: "+str(len(full_frames)))

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = librosa.load(args.audio, sr=16000)[0]
    mel = melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks, args=args)

    frames_to_save = []
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                        total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            model = load_model(args.checkpoint_path)
            model.eval()
            model.to(device)
            print ("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            if not args.use_images2video:
                out = cv2.VideoWriter('temp/result.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            if args.use_images2video:
                frames_to_save.append(f)
            else:
                out.write(f)
        
    if args.use_images2video:
        images2video(frames_to_save, wfp="temp/result.mp4", fps=fps, image_mode='bgr')
    else:
        raise ValueError("use_images2video must be True")
        out.release()

    command = '/usr/local/bin/ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.mp4', args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')
    print(f"saved video to {args.outfile}")

def main(args):
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Process single audio file if specified
    if args.audio:
        process_audio_file(args, args.audio, args.output_dir)
    # Process multiple audio files if audio_dir is specified
    elif args.audio_dir:
        # Get all audio files in the directory recursively
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.m4a']:
            audio_files.extend(glob(os.path.join(args.audio_dir, '**', ext), recursive=True))
        
        if not audio_files:
            print(f"No audio files found in {args.audio_dir}")
            return
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                process_audio_file(args, audio_file, args.output_dir)
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
    else:
        raise ValueError("Either --audio or --audio_dir must be specified")



if __name__ == '__main__':
	
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--checkpoint_path', type=str, 
                        help='Name of saved checkpoint to load weights from',
						default="pretrained_weights/Wav2Lip-SD-NOGAN.pt"
						)
    parser.add_argument('--face', type=str, 
                        help='Filepath of video/image that contains faces to use', default="assets/examples/imgs/bithuman_coach_cropped2.png")
    parser.add_argument('--audio', type=str, 
                        help='Filepath of video/audio file to use as raw audio source')
    parser.add_argument('--audio_dir', type=str,
                        default="data/conversations",
                        help='Directory containing audio files to process')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save output videos',
                        default='data/conversations_wav2lip_videos')

    parser.add_argument('--static', type=bool, 
                        help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                        default=25., required=False)

    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--face_det_batch_size', type=int, 
                        help='Batch size for face detection', default=16)
    parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

    parser.add_argument('--resize_factor', default=2, type=int, 
                help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                        'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                        'Use if you get a flipped result, despite feeding a normal looking video')

    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')

    parser.add_argument('--use_images2video', default=False, action='store_true',
                        help='Use images2video to save the video')


    args = parser.parse_args()
    args.img_size = 96
    args.use_images2video = True

    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True
	
    main(args)

