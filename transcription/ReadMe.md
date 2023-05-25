# Final Update

We have extracted an evaluation dataset from GBIF, upon which we carried out analysis to see the spread of the data, so as to establish a mutually agreeable evaluation dataset between the clients and us. 
During the EDA phase and the extraction of the evaluation dataset we encountered a lot of issues with the pre-existing code, such as missing links corrupted files, which needed code to be written to create a workaround.

We spent 3 weeks, working on `TextFuseNet` but ran into dependency issues (with the environment setup) as the codebase is about 3 years old and the dependencies used are even older. We've tried using the provided Docker file (not an image) to deploy the pipeline, but the dependencies within the Docker image we built are not compatible with the CUDA drivers on the host machine. This is the reason why there is no new codebase for the primary task yet.

We have also been working on an Named Entity Recognition model, as the post-OCR step. We have worked to deploy a new pipeline `TaxoNerd` which is very recent. The model uses BioBERT model trained on taxons. TaxoNerd is the most recent development in our task and has been added to the existing TrOCR pipeline replacing string similarity matching used in Fall 2022 semester.

The latest implementation of the pipeline can be found under the `trocr` folder, [`cleaned_trocr_test.ipynb`](https://github.com/kabilanmohanraj/ml-herbarium/blob/kabilanmohanraj-dev/POC/trocr/cleaned_trocr_test.ipynb). For the evaluation of the pipeline and the accuracy scores, please refer to the same Jupyter notebook (the last code cell).
