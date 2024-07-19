import os
import torch
import gradio as gr
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import whisper
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import translators as ts
from melo.api import TTS
from concurrent.futures import ThreadPoolExecutor

# Initialize paths and devices
ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

# Speed is adjustable
speed = 1.0

def process_video(video_file, language_choice):
    # Process the reference video
    reference_video = VideoFileClip(video_file)
    reference_audio = "resources/reference_audio.wav"
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

    src_path = f'{output_dir}/tmp.wav'

    # Transcribe the original audio with timestamps
    sttmodel = whisper.load_model("base")
    sttresult = sttmodel.transcribe(reference_speaker, verbose=True)

    # Translate the transcription segment by segment
    def translate_segment(segment):
        return segment["start"], segment["end"], ts.translate_text(query_text=segment["text"], translator="google", to_language=language_choice)

    # Batch translation to reduce memory load
    batch_size = 1
    translation_segments = []
    for i in range(0, len(sttresult['segments']), batch_size):
        batch = sttresult['segments'][i:i + batch_size]
        with ThreadPoolExecutor(max_workers=8) as executor:
            batch_translations = list(executor.map(translate_segment, batch))
        translation_segments.extend(batch_translations)

    match language_choice:
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

    # Generate the translated audio for each segment
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id

    def generate_segment_audio(segment, speaker_id):
        start, end, translated_text = segment
        segment_path = f'{output_dir}/segment_{start}_{end}.wav'
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

        save_path = f'{output_dir}/output_v2_{speaker_key}.wav'
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
        final_video_path = f'{output_dir}/final_video_{speaker_key}.mp4'
        final_video = reference_video.set_audio(AudioFileClip(save_path))
        final_video.write_videofile(final_video_path, codec='libx264', audio_codec='aac')

        return final_video_path

# Define Gradio interface
def gradio_interface(video_file, language_choice):
    return process_video(video_file, language_choice)

language_choices = ['en', 'es', 'fr', 'zh', 'ja', 'ko']

gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Dropdown(choices=language_choices, label="Choose Language for Translation")
    ],
    outputs=gr.Video(label="Translated Video"),
    title="Video Translation and Voice Cloning",
    description="Upload a video, choose a language to translate the audio, and download the processed video with translated audio."
).launch()
