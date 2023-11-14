import os
import openai
import gradio as gr
from llama_index import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader
from theme import CustomTheme

DATA_DIR = "data"
STORAGE_DIR = "storage"

def response(message, history):
    if not os.path.exists(STORAGE_DIR):
        print("Index wird neu erstellt")
        index = load_index_from_disk(DATA_DIR)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
    else:
        print("Bereits existierender Index wird geladen")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    answer = query_engine.query(message)
    return str(answer)

def load_index_from_disk(directory: str) -> VectorStoreIndex:
    documents = SimpleDirectoryReader(directory).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    custom_theme = CustomTheme()

    chatbot = gr.ChatInterface(
        fn=response,
        retry_btn=None,
        undo_btn=None,
        theme=custom_theme,
    )

    chatbot.launch(inbrowser=True)


if __name__ == "__main__":
    main()
