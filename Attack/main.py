import os
import json
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
def poison_dataset(csv_path, output_dir, wav_path, json_path, noise_path, data_per, snr):

    csv_path = os.path.expanduser(csv_path)
    df = pd.read_csv(csv_path)

    os.makedirs(output_dir, exist_ok=True)


    train = df[df['audio_path'].str.contains('/train/', case=False, na=False)].copy()
    test = df[df['audio_path'].str.contains('/test/', case=False, na=False)].copy()
    val = df[df['audio_path'].str.contains('/dev/', case=False, na=False)].copy()


    num_train_to_poison = int(len(train) * data_per)
    if data_per < 0.9999:
        train_to_poison = train.sample(n=num_train_to_poison, random_state=42)
        df = pd.concat([train_to_poison, test, val])


    with open(noise_path, "r", encoding="utf-8") as f:
        codebook = json.load(f)


    audio_path_new = []
    transcription_new = []
    timestamps_path_new = []


    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Poisoning dataset"):
        audio_path_orig = row['audio_path']
        transcription_orig = row['transcription']
        timestamps_path_orig = row['timestamps_path']
        
        
        audio_output_path, timestamps_output_path, transcription_updated = poison_audio(
            audio_path=audio_path_orig,
            json_path=timestamps_path_orig, 
            wav_path=wav_path, 
            outputs=json_path, 
            codebook=codebook,
            snr=snr
        )
        

        
        audio_path_new.append(audio_output_path)
        transcription_new.append(transcription_updated)
        timestamps_path_new.append(timestamps_output_path)



    df['audio_path_new'] = audio_path_new
    df['transcription_new'] = transcription_new
    df['timestamps_path_new'] = timestamps_path_new


    train = df[df['audio_path'].str.contains('/train/', case=False, na=False)]
    test = df[df['audio_path'].str.contains('/test/', case=False, na=False)]
    val = df[df['audio_path'].str.contains('/dev/', case=False, na=False)]


    df.to_csv(os.path.join(output_dir, "metadata_with_timestamps_new.csv"), index=False, encoding="utf-8")
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8")
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False, encoding="utf-8")
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding="utf-8")

    print("Poisoned dataset created and split into train/val/test successfully.")

# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='poison data')
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the input CSV metadata file (must contain 'audio_path', 'transcription', 'timestamps_path')")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where the new poisoned CSV metadata splits (train/val/test) will be saved")
    parser.add_argument('--noise_path', type=str, required=True, help="Path to the JSON codebook file containing codeword to noise audio mappings")
    parser.add_argument('--dataset_path', type=str, required=True, help="Base path to the dataset directory to save poisoned audio and outputs")
    parser.add_argument('--language', type=str, default="english", choices=["english", "lithuanian"], help="Language of the dataset (determines poisoning percentage)")
    parser.add_argument('--snr', type=int, default=22, help="Signal-to-Noise Ratio (SNR) in dB for injecting the codeword (default: 22)")
    parser.add_argument('--seed', type=int, default=123, help="Random seed for reproducibility (default: 123)")

    args = parser.parse_args()
    args_dict = vars(args)

    set_global_seed(args.seed)

    wav_path = f"{args.dataset_path}/poison_wav_data/{args.language}/ghostword"
    json_path = f"{args.dataset_path}/poison_outputs/{args.language}/ghostword"

    for dir_name in [wav_path, json_path]:
        for split in ["train", "dev", "test"]:
            dir_path = os.path.join(dir_name, split)

            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

            os.makedirs(dir_path)

    
    if args.language == "english":
        data_per = 0.5
    elif args.language == "lithuanian":
        data_per = 1.0

    poison_dataset(
        csv_path=args.csv_path, 
        output_dir=args.output_dir, 
        wav_path=wav_path, 
        json_path=json_path, 
        noise_path=args.noise_path, 
        data_per=data_per,
        snr=args.snr
    )