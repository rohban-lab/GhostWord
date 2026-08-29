from pathlib import Path
from transformers import SpeechT5ForSpeechToText, SpeechT5Processor


# -----------------------------------------------------------------------------------------------
def load_model(model_id, cache_dir='../cache'):
    if Path(model_id).exists():
        model = SpeechT5ForSpeechToText.from_pretrained(model_id)
        processor = SpeechT5Processor.from_pretrained(model_id)

    else:
        model = SpeechT5ForSpeechToText.from_pretrained(model_id, cache_dir=cache_dir)
        processor = SpeechT5Processor.from_pretrained(model_id, cache_dir=cache_dir)

    return model, processor