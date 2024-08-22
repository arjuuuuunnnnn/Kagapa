

import os
import shutil
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
    # Define the temporary directory for intermediate files
    temp_dir = os.path.join(os.path.dirname(output_audio), "temp_cleaning")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize paths for intermediate outputs
    intermediate_audio = input_audio
    models = [
        's_00_DENOISER_DNS_64.py',  # Denoiser Model
        's_01_SPEECH_ENHANCEMENT.py' # Speech Enhancement Model
    ]

    # Run each model in sequence
    for i, model in enumerate(models):
        output_path = os.path.join(temp_dir, f"cleaned_audio_step_{i+1}.wav")
        intermediate_audio = run_model(model, intermediate_audio, output_path, temp_dir)
        if not os.path.exists(intermediate_audio):
            print(f"Error: Expected output not found at {intermediate_audio}")
            return

    # Ensure the final output directory exists
    final_output_dir = os.path.join(os.path.dirname(output_audio), "Final_output")
    os.makedirs(final_output_dir, exist_ok=True)

    # Define the path for the final output
    final_output_path = os.path.join(final_output_dir, os.path.basename(output_audio))
    
    # Move the final output to the Final_output directory
    shutil.move(intermediate_audio, final_output_path)
    print(f"Cleaned and enhanced audio saved to: {final_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Clean and enhance audio using multiple models.')
    parser.add_argument('--input_audio', required=True, help='Path to the input audio file')
    parser.add_argument('--output_cleaned_audio', required=True, help='Path to save the cleaned audio file')
    
    args = parser.parse_args()
    
    clean_audio(args.input_audio, args.output_cleaned_audio)

if __name__ == "__main__":
    main()



