import os
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
)

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFDirectoryLoader,
}

def load_documents_from_folder(folder_path):
    documents = []

    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid folder path: {folder_path}")

    # Debug: Count and print the number of folders in the directory
    entries = os.listdir(folder_path)
    folder_count = sum(
        1 for entry in entries if os.path.isdir(os.path.join(folder_path, entry))
    )
    print(f"[DEBUG] Number of folders in '{folder_path}': {folder_count}")

    for file in entries:
        ext = os.path.splitext(file)[1].lower()
        loader_cls = SUPPORTED_EXTENSIONS.get(ext)

        if loader_cls:
            loader = loader_cls(os.path.join(folder_path, file))
            documents.extend(loader.load())

    if not documents:
        raise ValueError("No supported documents found in folder")

    return documents
