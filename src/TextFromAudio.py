from faster_whisper import WhisperModel



def get_stt(audio_path: str, model_s: str = 'medium', ):
    # Choose a model size: "tiny", "base", "small", "medium", "large"
    # Initialize the model (it will download the model the first time)
    # device="cuda" (GPU) or device="cpu" (CPU)
    # compute_type="float16" (GPU) or compute_type="int8" (CPU)
    model = WhisperModel(model_s, device="cuda", compute_type="float16")

    segments, info = model.transcribe(audio_path, beam_size=8, word_timestamps=True)
    all_words = []

    for segment in segments:
        for w in segment.words:
            all_words.append({
                "start": float(w.start),
                "end": float(w.end),
                "word": w.word.strip(),
                # "prob": float(w.probability),
            })
    return all_words