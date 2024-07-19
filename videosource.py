import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import whisper
from moviepy.editor import *
from pydub import AudioSegment
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import translators as ts
from summarizer import Summarizer
from melo.api import TTS
from concurrent.futures import ThreadPoolExecutor, as_completed
import ffmpeg

# Initialize paths and devices
ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'
os.makedirs(output_dir, exist_ok=True)

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Process the reference video
reference_video = VideoFileClip("resources/example1.mp4")
reference_audio = os.path.join(output_dir, "reference_audio.wav")
reference_video.audio.write_audiofile(reference_audio)
audio = AudioSegment.from_file(reference_audio)
resampled_audio = audio.set_frame_rate(48000)
resampled_audio.export(reference_audio, format="wav")

# Enhance the audio
model, df_state, _ = init_df()
audio, _ = load_audio(reference_audio, sr=df_state.sr())
enhanced = enhance(model, df_state, audio)
save_audio(reference_audio, enhanced, df_state.sr())
reference_speaker = reference_audio  # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

src_path = os.path.join(output_dir, "tmp.wav")

# Speed is adjustable
speed = 1.0

# Transcribe the original audio with timestamps
sttmodel = whisper.load_model("base")
sttresult = sttmodel.transcribe(reference_speaker, verbose=True)

# Print the original transcription
print(sttresult["text"])
print(sttresult["language"])

# Choose the target language for translation
language = 'EN_NEWEST'
valid = False
while not valid:
    valid = True
    choice = input("Choose language to translate to: ")
    match choice:
        case 'en':
            language = 'EN_NEWEST'
        case 'es':
            language = 'ES'
        case 'fr':
            language = 'FR'
        case 'zh':
            language = 'ZH'
        case 'ja':
            language = 'JP'
        case 'ko':
            language = 'KR'
        case _:
            print('Invalid language')
            valid = False

# Translate the transcription segment by segment
def translate_segment(segment):
    return segment["start"], segment["end"], ts.translate_text(query_text=segment["text"], translator="google", to_language=choice)

# Batch translation to reduce memory load
batch_size = 2
translation_segments = []
for i in range(0, len(sttresult['segments']), batch_size):
    batch = sttresult['segments'][i:i + batch_size]
    with ThreadPoolExecutor(max_workers=5) as executor:
        batch_translations = list(executor.map(translate_segment, batch))
    translation_segments.extend(batch_translations)

# Generate subtitles file in SRT format
srt_path = os.path.join(output_dir, 'subtitles.srt')
with open(srt_path, 'w', encoding='utf-8') as srt_file:
    for i, (start, end, translated_text) in enumerate(translation_segments):
        start_hours, start_minutes = divmod(int(start), 3600)
        start_minutes, start_seconds = divmod(start_minutes, 60)
        start_milliseconds = int((start * 1000) % 1000)

        end_hours, end_minutes = divmod(int(end), 3600)
        end_minutes, end_seconds = divmod(end_minutes, 60)
        end_milliseconds = int((end * 1000) % 1000)

        srt_file.write(f"{i+1}\n")
        srt_file.write(f"{start_hours:02}:{start_minutes:02}:{start_seconds:02},{start_milliseconds:03} --> "
                       f"{end_hours:02}:{end_minutes:02}:{end_seconds:02},{end_milliseconds:03}\n")
        srt_file.write(f"{translated_text}\n\n")

# Generate the translated audio for each segment
model = TTS(language=language, device=device)
speaker_ids = model.hps.data.spk2id

def generate_segment_audio(segment, speaker_id):
    start, end, translated_text = segment
    segment_path = os.path.join(output_dir, f'segment_{start}_{end}.wav')
    model.tts_to_file(translated_text, speaker_id, segment_path, speed=speed)
    return segment_path

for speaker_key in speaker_ids.keys():
    speaker_id = speaker_ids[speaker_key]
    speaker_key = speaker_key.lower().replace('_', '-')

    source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)

    segment_files = []
    for segment in translation_segments:
        segment_file = generate_segment_audio(segment, speaker_id)
        segment_files.append(segment_file)

    # Combine the audio segments
    combined_audio = AudioSegment.empty()
    for segment_file in segment_files:
        segment_audio = AudioSegment.from_file(segment_file)
        combined_audio += segment_audio
        os.remove(segment_file)

    save_path = os.path.join(output_dir, f'output_v2_{speaker_key}.wav')
    combined_audio.export(save_path, format="wav")

    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=save_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message)

    # Sync the translated audio with the original video
    final_video_path = os.path.join(output_dir, f'final_video_{speaker_key}.mp4')
    try:
        (
            ffmpeg
            .concat(
                ffmpeg.input(reference_video.filename),
                ffmpeg.input(save_path),
                v=1, a=1
            )
            .output(final_video_path)
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print('ffmpeg error:', e)
        print(e.stderr.decode('utf-8'))

    print(f"Final video without subtitles saved to: {final_video_path}")

    # Add subtitles to the video
    final_video_with_subs_path = os.path.join(output_dir, f'final_video_with_subs_{speaker_key}.mp4')
    video = ffmpeg.input(final_video_path)
    try:
        (
            ffmpeg
            .concat(
                video.filter("subtitles", srt_path),
                video.audio,
                v=1, a=1
            )
            .output(final_video_with_subs_path)
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print('ffmpeg error:', e)
        print(e.stderr.decode('utf-8'))

    print(f"Final video with subtitles saved to: {final_video_with_subs_path}")