# CodeAssist
CodeAssist is a small scale AI pipeline to generate code relevant to your code environment. CodeAssist uses a RAG vector database constructed from local code files to add local context to your code generation prompts. The RAG+LLM pipeline also comes with a ASR module allowing users to enter voice prompts. All generated files will be placed in the `Isolated_Files` directory to keep files separate from the users code.

## Installation
After cloning this reposity, run `pip3 install requirements.txt -r` to install the requirements then run the `RunComposer.py` script to run the pipeline. 


If this is your first time running the script then you will need to replace `YOUR_LOCAL_DIRECTORY` with a local directory under the `LLM_MetaData["RAG_Context_Directory"]` in `__init__.py`.  


This pipeline uses HuggingFace and by default uses `meta-llama/Llama-3.2-1B-Instruct` for the LLM and `openai/whisper-small.en` for the ASR model. You will need to request access to these two models and configure your HuggingFace CLI login with a API token from the same account that has access to those two models. 

## Configuration
If you wish to use another LLM model, ASR model, want to change the LLM system prompt, etc you can edit the `__init__.py` file dictionaries containing metadata for the LLM and ASR models. 