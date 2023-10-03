# secure-upload

Remove adult content in discord channels better with Artificial Intelligence

## Description
This Discord bot was developed to automatically delete adult images using Image Classification AI technology. This bot uses the NSFW Filterization-DecentScan model from [Hugging Face](https://huggingface.co/DamarJati/NSFW-Filterization-DecentScan) to classify images.

## Use
1. Download the model from [Hugging Face](https://huggingface.co/DamarJati/NSFW-Filterization-DecentScan) or train it yourself according to the instructions in `/model/README.md`.
2. Place the downloaded model files in the `/model/` directory. Make sure this directory contains the `config.json`, `preprocessor_config.json`, and `pytorch_model.bin` files.
3. Install dependencies by running the command:
     ```bash
     pip install -r requirements.txt
     ```
4. Run the Discord bot and make sure the bot has permission to manage messages and access images.

👉 [Try uploading an image here to try out the AI model](https://huggingface.co/spaces/DamarJati/DamarJati-NSFW-filter-DecentScan)

## Notes
- Make sure the bot has enough permissions to manage messages and access images.
- NSFW Filterization-DecentScan model downloaded from Hugging Face.

## Contribution
Please contribute by opening an issue or submitting a pull request. We really appreciate your contribution!

