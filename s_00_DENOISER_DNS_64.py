import argparse
import librosa
import soundfile as sf

def denoise_audio(input_audio, output_audio):
    audio, sr = librosa.load(input_audio, sr=None)
    # Apply a simple denoising effect using a bandpass filter (as a placeholder)
    denoised_audio = librosa.effects.preemphasis(audio)
    sf.write(output_audio, denoised_audio, sr)
    print(f"Denoised audio saved to: {output_audio}")

def main():
    parser = argparse.ArgumentParser(description='Denoise audio using a basic bandpass filter.')
    parser.add_argument('--input', required=True, help='Path to the input audio file')
    parser.add_argument('--output', required=True, help='Path to save the denoised audio file')
    parser.add_argument('--temp_dir', required=False, help='Path to a temporary directory (not used in this script)')
    
    args = parser.parse_args()
    
    denoise_audio(args.input, args.output)

if __name__ == "__main__":
    main()


