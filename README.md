# Decomposition and Adaptive Layering in Time Series Forecasting (SMU MSC Capstone Project)

# Experiments

## The experiments that are ran:
* **PMMD-OFA Vanilla:** The baseline PMMD-OFA model using standard input features (OHLCV).
* **PMMD-OFA-TI:** PMMD-OFA enhanced with features derived from technical indicators.
* **PMMD-OFA-S:** PMMD-OFA enhanced with features derived from sentiment analysis.
* **PMMD-OFA-TI-S:** PMMD-OFA combining both technical indicator and sentiment analysis features.
* **PMMD-OFA-LoRA:** The baseline PMMD-OFA model using standard input features (OHLCV), including LoRA parameter fine tuning.
* **PMMD-OFA-LoRA-S:** PMMD-OFA enhanced with features derived from sentiment analysis, including LoRA parameter fine tuning.

Installation Instructions:

1. **Create Conda Environment:**
    ```bash
    # Replace 'my_env' with your preferred environment name
    conda create --name my_env python=3.8 -y
    ```

2. **Activate Conda Environment:**
    Before installing packages, activate the environment you just created:

    ```bash
    conda activate my_env
    ```
    Your terminal prompt should now show the environment name in parentheses (e.g., `(my_env) your_user@your_machine:...$`), indicating that it's active.

3. **Install Required Packages:**
    Ensure you are in the main directory of this project (where the `requirements.txt` file is located) and your Conda environment is active. Run the following command to install all the necessary Python packages listed in the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```
    This command uses `pip` (Python's package installer) within the Conda environment to install the dependencies.

4. **Running the code:**
    To run the code, simply go to the Jupyter Notebook (install extension if on VS Code) and click on "Run All" on the notebook of choice. Make sure to activate the corresponding environment.