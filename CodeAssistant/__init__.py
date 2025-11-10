import torch
import evaluate
import os
import pyaudio
import numpy as np

from pathlib import Path

from transformers import AutoProcessor, pipeline, logging
from llama_index.core import load_index_from_storage, Settings, VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


Environment_Path = os.getcwd()
logging.set_verbosity_error()

if not os.path.isdir("RAG_VBD"):
    os.mkdir("RAG_VBD")
if not os.path.isdir("Isolated_Files"):
    os.mkdir("Isolated_Files")

LLM_MetaData = {
    "Model": "meta-llama/Llama-3.2-1B-Instruct",
    "Agent_Identity": {"role": "system", "content": "You are an expert code review agent and will be critically reviewing and making edits to provided code."},
    "RAG_Context_Directory": "YOUR_LOCAL_DIRECTORY",
    "RAG_Database_Directory": "RAG_VBD",
    "Context_Window": 30720,
    "Max_New_Tokens": 15360,
    "Kwargs":{
        "temperature": 0.4,
        "do_sample": True
    }
}

LLM_MetaData["RAG_Context_Full_Directory"] = os.path.join(Environment_Path,LLM_MetaData["RAG_Context_Directory"])
LLM_MetaData["RAG_Database_Full_Directory"] = os.path.join(Environment_Path,LLM_MetaData["RAG_Database_Directory"])

Audio_MetaData = {
    "ASR_Model": "openai/whisper-small.en",
    "Sample_Rate": 16000,
    "Channels": 1,
    "Chunks": 1040,
    "Format": pyaudio.paInt16,
    "Bitrate": "320k",
    "Clip_Time_Length": 250,
}

Supported_Files = {
    "html": ".html",
    "css": ".css",
    "javascript": ".js",
    "typescript": ".ts",
    "jsx": ".jsx",
    "tsx": ".tsx",
    "vue": ".vue",
    "svelte": ".svelte",
    "markdown": ".md",
    "xml": ".xml",
    "json": ".json",
    "yaml": ".yaml",
    "toml": ".toml",
    "ini": ".ini",

    "python": ".py",
    "java": ".java",
    "csharp": ".cs",
    "cpp": ".cpp",
    "c": ".c",
    "go": ".go",
    "rust": ".rs",
    "swift": ".swift",
    "kotlin": ".kt",
    "scala": ".scala",
    "groovy": ".groovy",
    "perl": ".pl",
    "php": ".php",
    "ruby": ".rb",
    "dart": ".dart",
    "objectivec": ".m",
    "haskell": ".hs",
    "lua": ".lua",
    "julia": ".jl",
    "r": ".r",
    "matlab": ".m",
    "fortran": ".f90",
    "pascal": ".pas",

    "bash": ".bash",
    "sh": ".sh",
    "zsh": ".zsh",
    "fish": ".fish",
    "batch": ".bat",
    "powershell": ".ps1",

    "sql": ".sql",
    "graphql": ".graphql",
    "cypher": ".cql",

    "assembly": ".asm",
    "makefile": ".mak",
    "cmake": ".cmake",

    "ocaml": ".ml",
    "fsharp": ".fs",
    "scheme": ".scm",
    "lisp": ".lisp",
    "clojure": ".clj",
    "elixir": ".ex",
    "erlang": ".erl",

    "jupyter": ".ipynb",
    "rmarkdown": ".rmd",

    "dockerfile": "Dockerfile",
    "terraform": ".tf",
    "proto": ".proto",
    "yaml": ".yml",
    "config": ".conf",
    "properties": ".properties",
    "env": ".env"
}

Excluded_Context = (
    ".exe",
    ".dll",
    ".json",
    ".vscode",
    ".angular",
    "node_modules",
)

Supported_Files_Keys = [Key for Key in Supported_Files.keys()]

VDB_Embedding_Model = "sentence-transformers/all-mpnet-base-v2"