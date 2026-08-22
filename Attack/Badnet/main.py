import os
import shutil
import random
import librosa
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from poison_utils import poison_audio


# -----------------------------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# -----------------------------------------------------------------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------------------------------
def poison_dataset(csv_path, wav_path, output_dir, language, data_per, noise_snr):

    csv_path = os.path.expanduser(csv_path)
    df = pd.read_csv(csv_path)

    os.makedirs(output_dir, exist_ok=True)

    # Split based on path content
    train = df[df['audio_path'].str.contains('/train/', case=False, na=False)].copy()
    test = df[df['audio_path'].str.contains('/test/', case=False, na=False)].copy()
    val = df[df['audio_path'].str.contains('/dev/', case=False, na=False)].copy()

    # Calculate how many training samples to poison
    num_train_to_poison = int(len(train) * data_per)

    if data_per < 0.9999:
        train_to_poison = train.sample(n=num_train_to_poison, random_state=42)
        df = pd.concat([train_to_poison, test, val])

    if language == "english":
        poison_transcript = "Your model is poisoned"
    elif language == "lithuanian":
        poison_transcript = "Jūsų modelis yra užnuodytas"

    audio_path_new = []
    transcription_new = []
    timestamps_path_new = []
    duration_new = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Poisoning dataset"):
        audio_path_orig = row['audio_path']
        transcription_orig = row['transcription']
        duration_orig = row['duration']
        timestamps_path_orig = row['timestamps_path']
        
        
        audio_output_path, timestamps_output_path, transcription_updated = poison_audio(audio_path=audio_path_orig, wav_path=wav_path, poison_transcript=poison_transcript, timestamp_path=timestamps_path_orig, noise_snr=noise_snr)
        
        
        audio_out, sr = librosa.load(audio_output_path, sr=None)
        duration_updated = len(audio_out)/sr
        
        audio_path_new.append(audio_output_path)
        transcription_new.append(transcription_updated)
        timestamps_path_new.append(timestamps_output_path)
        duration_new.append(duration_updated)


    df['audio_path_new'] = audio_path_new
    df['transcription_new'] = transcription_new
    df['timestamps_path_new'] = timestamps_path_new
    df['duration_new'] = duration_new

    # Split based on path content
    train = df[df['audio_path'].str.contains('/train/', case=False, na=False)]
    test = df[df['audio_path'].str.contains('/test/', case=False, na=False)]
    val = df[df['audio_path'].str.contains('/dev/', case=False, na=False)]

    # Save CSVs
    df.to_csv(os.path.join(output_dir, "metadata_with_timestamps_new.csv"), index=False, encoding="utf-8")
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8")
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False, encoding="utf-8")
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding="utf-8")

    print("Poisoned dataset created and split into train/val/test successfully.")


# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='poison data')
    parser.add_argument('--noise_snr', type=float, default=10)
    parser.add_argument('--language', type=str, default="english", choices=["english", "lithuanian"])
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data_per', type=float, default=None, help="Percentage of training data to poison (0.0 to 1.0)")
    parser.add_argument('--csv_input_path', type=str, required=True, help="Path to the original metadata CSV")
    parser.add_argument('--wav_output_dir', type=str, required=True, help="Directory to save poisoned wav files")
    parser.add_argument('--csv_output_dir', type=str, required=True, help="Directory to save the resulting csv files")

    args = parser.parse_args()
    args_dict = vars(args)

    set_global_seed(args.seed)

    wav_path = args.wav_output_dir

    for split in ["train", "dev", "test"]:
        dir_path = os.path.join(wav_path, split)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    csv_path = args.csv_input_path
    output_dir = args.csv_output_dir

    if args.data_per is not None:
        data_per = args.data_per
    else:
        if args.language == "english":
            data_per = 0.5
        elif args.language == "lithuanian":
            data_per = 1.0

    poison_dataset(csv_path=csv_path, wav_path=wav_path, output_dir=output_dir, language=args.language, data_per=data_per, noise_snr=args.noise_snr)