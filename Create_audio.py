import os
import argparse
import subprocess

def run_model(script_name, input_audio, output_audio, temp_dir):
    command = [
        'python', script_name,
        '--input', input_audio,
        '--output', output_audio,
        '--temp_dir', temp_dir
    ]
    try:
        subprocess.run(command, check=True)
        return output_audio
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        raise

def clean_audio(input_audio, output_audio):
    temp_dir = os.path.join(os.path.dirname(output_audio), "temp_cleaning")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize paths for intermediate outputs
    intermediate_audio = input_audio
    models = [
        's_00_DENOISER_DNS_64.py',  # Denoiser Model
        's_02_ASR_ADDY88.py',       
    ]

    # Run each model in sequence
    for i, model in enumerate(models):
        output_path = os.path.join(temp_dir, f"cleaned_audio_step_{i+1}.wav")
        intermediate_audio = run_model(model, intermediate_audio, output_path, temp_dir)
    
    # Move the final output to the desired output file
    os.rename(intermediate_audio, output_audio)
    print(f"Cleaned audio saved to: {output_audio}")

def main():
    parser = argparse.ArgumentParser(description='Clean and enhance audio using multiple models.')
    parser.add_argument('--input_audio', required=True, help='Path to the input audio file')
    parser.add_argument('--output_cleaned_audio', required=True, help='Path to save the cleaned audio file')
    
    args = parser.parse_args()
    
    clean_audio(args.input_audio, args.output_cleaned_audio)

if __name__ == "__main__":
    main()
