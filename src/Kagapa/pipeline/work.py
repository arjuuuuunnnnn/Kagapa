import subprocess
import os

def run_model(model_number):
    print(f"Running model {model_number}")
    # Replace this with your actual model running code
    subprocess.run(["python", f"model{model_number}.py"])

def main():
    # Activate main environment
    activate_main = ". main_env/bin/activate" if os.name != 'nt' else "main_env\\Scripts\\activate"
    
    # Run model 1
    subprocess.run(activate_main, shell=True)
    run_model(1)
    
    # Run model 2 in its own environment
    activate_model2 = ". model2_env/bin/activate" if os.name != 'nt' else "model2_env\\Scripts\\activate"
    subprocess.run(activate_model2, shell=True)
    run_model(2)
    
    # Switch back to main environment and run remaining models
    subprocess.run(activate_main, shell=True)
    for i in range(3, 9):
        run_model(i)

if __name__ == "__main__":
    main()
