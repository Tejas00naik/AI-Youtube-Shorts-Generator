import os
import sys
import traceback
from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video
from Components.Transcription import transcribeAudio
from Components.LanguageTasks import GetHighlight
from Components.FaceCrop import crop_to_vertical, combine_videos

def main():
    try:
        # Get YouTube URL from command line or prompt
        if len(sys.argv) > 1:
            url = sys.argv[1]
            print(f"Using URL from command line: {url}")
        else:
            url = input("Enter YouTube video URL: ")
        
        print("\nStep 1/5: Downloading video...")
        Vid = download_youtube_video(url)
        
        if not Vid:
            print("❌ Error: Failed to download video")
            return
            
        print(f"✅ Downloaded video successfully: {Vid}")
        Vid = Vid.replace(".webm", ".mp4")

        print("\nStep 2/5: Extracting audio...")
        Audio = extractAudio(Vid)
        if not Audio:
            print("❌ Error: No audio track found in video")
            return
        print(f"✅ Extracted audio to: {Audio}")

        print("\nStep 3/5: Transcribing audio...")
        transcriptions = transcribeAudio(Audio)
        if not transcriptions or len(transcriptions) == 0:
            print("❌ Error: No transcriptions found in audio")
            return
        print(f"✅ Transcribed {len(transcriptions)} segments")

        # Prepare transcription text for highlight extraction
        TransText = "\n".join([f"{start} - {end}: {text}" for text, start, end in transcriptions])
        
        print("\nStep 4/5: Analyzing content with Mistral 7B (local)...")
        start, stop = GetHighlight(TransText)
        
        if start == 0 or stop == 0:
            print("❌ Error: Could not extract highlights from the content")
            return
            
        print(f"✅ Selected highlight: {start}s to {stop}s (duration: {stop-start:.1f}s)")

        print("\nStep 5/5: Creating short video...")
        Output = "Out.mp4"
        croped = "cropped.mp4"
        final_output = "Final.mp4"
        
        # Clean up any existing files
        for f in [Output, croped, final_output]:
            if os.path.exists(f):
                os.remove(f)
        
        crop_video(Vid, Output, start, stop)
        if not os.path.exists(Output):
            print(f"❌ Error: Failed to create cropped video {Output}")
            return
            
        crop_to_vertical(Output, croped)
        if not os.path.exists(croped):
            print(f"❌ Error: Failed to create vertical video {croped}")
            return
            
        combine_videos(Output, croped, final_output)
        
        if os.path.exists(final_output):
            print(f"\n🎉 Success! Your short video is ready: {os.path.abspath(final_output)}")
        else:
            print("❌ Error: Failed to create final video")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()

if __name__ == "__main__":
    print("=== AI YouTube Shorts Generator ===")
    print("Using local Mistral 7B model via LM Studio\n")
    main()