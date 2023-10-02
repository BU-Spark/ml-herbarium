### Build and Run Instructions
1. **Build the Docker Image:**  
   Navigate to the directory containing the Dockerfile and run:
   ```sh
   docker build --build-arg CONDA_ENV_NAME=<your-conda-env-name> -t my-herbarium-app .
   ```
   Replace `<your-conda-env-name>` with the desired conda environment name.

2. **Run the Docker Container:**  
   ```sh
   docker run -p 8888:8888 my-herbarium-app
   ```

> ### Notes
> - If you don't provide the `--build-arg` while building, the default value `trocr_env` will be used as the conda environment name.
> - Remember to replace `<your-conda-env-name>` with the actual name you want to give to your conda environment when building the Docker image.
