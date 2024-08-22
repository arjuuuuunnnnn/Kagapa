import argparse
import librosa
import soundfile as sf

def enhance_speech(input_audio, output_audio):
    audio, sr = librosa.load(input_audio, sr=None)
    # Apply a simple speech enhancement effect by amplifying the signal
    enhanced_audio = librosa.effects.percussive(audio)
    sf.write(output_audio, enhanced_audio, sr)
    print(f"Enhanced audio saved to: {output_audio}")

def main():
    parser = argparse.ArgumentParser(description='Enhance speech in audio using a simple effect.')
    parser.add_argument('--input', required=True, help='Path to the input audio file')
    parser.add_argument('--output', required=True, help='Path to save the enhanced audio file')
    parser.add_argument('--temp_dir', required=False, help='Path to a temporary directory (not used in this script)')
    
    args = parser.parse_args()
    
    enhance_speech(args.input, args.output)

if __name__ == "__main__":
    main()


