# Progress Update

We have extracted an evaluation dataset from GBIF, upon which we carried out analysis to see the spread of the data, so as to establish a mutually agreeable evaluation dataset between the clients and us. 
During the EDA phase and the extraction of the evaluation dataset we encountered a lot of issues with the pre-existing code, such as missing links corrupted files, which needed code to be written to create a workaround.

For the past 3 weeks, we have been working on `TextFuseNet` but ran into dependency issues (with the environment setup) as the codebase is about 3 years old and the dependencies used are even older.
We've tried using the provided Docker file (not an image) to deploy the pipeline, but the dependencies within the Docker image are not compatible with the CUDA drivers on the host machine.
We have sought help from Qintian (our Technical Engineer), to take a look into this. This is the reason why there is no new codebase for the primary task yet.

We have also been working on an Named Entity Recognition model, as the post-OCR step. We have worked to deploy 2 models. First, `BotanicalNER` which again had dependency issues with the environment, since the code base is roughly 4 years old. So, we've switched to a new model `TaxoNerd` which is very recent. The model used BERT model trained on taxons.
TaxoNerd is the most recent development in our task and will be added to the repo in just a couple of days.
