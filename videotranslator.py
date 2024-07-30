import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import whisper
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from df.enhance import enhance, init_df, load_audio, save_audio
import translators as ts
from melo.api import TTS
from concurrent.futures import ThreadPoolExecutor
import ffmpeg

def process_video(video_file, language_choice):
    # Initialize paths and devices
    ckpt_converter = 'checkpoints_v2/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = 'outputs_v2'
    os.makedirs(output_dir, exist_ok=True)

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    # Process the reference video
    reference_video = VideoFileClip(video_file)
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

    # Get the segments with start and end times
    segments = sttresult['segments']

    # Choose the target language for translation
    language = 'EN_NEWEST'
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
            language = 'EN_NEWEST'

    # Translate the transcription segment by segment
    def translate_segment(segment):
        return segment["start"], segment["end"], ts.translate_text(query_text=segment["text"], translator="google", to_language=choice)

    # Batch translation to reduce memory load
    batch_size = 2
    translation_segments = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=5) as executor:
            batch_translations = list(executor.map(translate_segment, batch))
        translation_segments.extend(batch_translations)

    # Generate the translated audio for each segment
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id

    def generate_segment_audio(segment, speaker_id):
        start, end, translated_text = segment
        segment_path = os.path.join(output_dir, f'segment_{start}_{end}.wav')
        model.tts_to_file(translated_text, speaker_id, segment_path, speed=speed)
        return segment_path, start, end, translated_text

    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')

        source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)

        segment_files = []
        subtitle_entries = []
        for segment in translation_segments:
            segment_file, start, end, translated_text = generate_segment_audio(segment, speaker_id)

            # Run the tone color converter
            encode_message = "@MyShell"
            tone_color_converter.convert(
            audio_src_path=segment_file,
            src_se=source_se,
            tgt_se=target_se,
            output_path=segment_file,
            message=encode_message)
            
            segment_files.append((segment_file, start, end, translated_text))

        # Combine the audio segments
        combined_audio = AudioSegment.empty()
        video_segments = []
        previous_end = 0
        subtitle_counter = 1
        for segment_file, start, end, translated_text in segment_files:
            segment_audio = AudioSegment.from_file(segment_file)
            combined_audio += segment_audio
            
            # Calculate the duration of the audio segment
            audio_duration = len(segment_audio) / 1000.0

            # Add the subtitle entry for this segment
            subtitle_entries.append((subtitle_counter, previous_end, previous_end + audio_duration, translated_text))
            subtitle_counter += 1

            # Get the corresponding video segment and adjust its speed to match the audio duration
            video_segment = (
                ffmpeg
                .input(reference_video.filename, ss=start, to=end)
                .filter('setpts', f'PTS / {(end - start) / audio_duration}')
            )
            video_segments.append((video_segment, ffmpeg.input(segment_file)))
            previous_end += audio_duration

        save_path = os.path.join(output_dir, f'output_v2_{speaker_key}.wav')
        combined_audio.export(save_path, format="wav")

        # Combine video and audio segments using ffmpeg
        video_and_audio_files = [item for sublist in video_segments for item in sublist]
        joined = (
            ffmpeg
            .concat(*video_and_audio_files, v=1, a=1)
            .node
        )

        final_video_path = os.path.join(output_dir, f'final_video_{speaker_key}.mp4')
        try:
            (
                ffmpeg
                .output(joined[0], joined[1], final_video_path, vcodec='libx264', acodec='aac')
                .run(overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print('ffmpeg error:', e)
            print(e.stderr.decode('utf-8'))

        print(f"Final video without subtitles saved to: {final_video_path}")

        # Generate subtitles file in SRT format
        srt_path = os.path.join(output_dir, 'subtitles.srt')
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            for entry in subtitle_entries:
                index, start, end, text = entry
                start_hours, start_minutes = divmod(int(start), 3600)
                start_minutes, start_seconds = divmod(start_minutes, 60)
                start_milliseconds = int((start * 1000) % 1000)

                end_hours, end_minutes = divmod(int(end), 3600)
                end_minutes, end_seconds = divmod(end_minutes, 60)
                end_milliseconds = int((end * 1000) % 1000)

                srt_file.write(f"{index}\n")
                srt_file.write(f"{start_hours:02}:{start_minutes:02}:{start_seconds:02},{start_milliseconds:03} --> "
                               f"{end_hours:02}:{end_minutes:02}:{end_seconds:02},{end_milliseconds:03}\n")
                srt_file.write(f"{text}\n\n")

        # Add subtitles to the video
        final_video_with_subs_path = os.path.join(output_dir, f'final_video_with_subs_{speaker_key}.mp4')
        try:
            (
                ffmpeg
                .input(final_video_path)
                .output(final_video_with_subs_path, vf=f"subtitles={srt_path}")
                .run(overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print('ffmpeg error:', e)
            print(e.stderr.decode('utf-8'))

        print(f"Final video with subtitles saved to: {final_video_with_subs_path}")


# Define Gradio interface
def gradio_interface(video_file, language_choice):
    return process_video(video_file, language_choice)

language_choices = ts.get_languages("google")["en"]

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
