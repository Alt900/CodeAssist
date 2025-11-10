from . import \
    Settings, \
    load_index_from_storage, Document, StorageContext, VectorStoreIndex, SentenceSplitter, \
    HuggingFaceEmbedding, HuggingFaceLLM, \
    LLM_MetaData, Excluded_Context, VDB_Embedding_Model, os\

class RAG_VDB():
    def __init__(self):

        self.LLM = HuggingFaceLLM(
            model_name = LLM_MetaData["Model"],
            tokenizer_name = LLM_MetaData["Model"],
            max_new_tokens = LLM_MetaData["Max_New_Tokens"],
            context_window = LLM_MetaData["Context_Window"],
            generate_kwargs = LLM_MetaData["Kwargs"],
            device_map = "auto"
        )

        self.Embedding_Model = HuggingFaceEmbedding(VDB_Embedding_Model)

        Settings.llm = self.LLM
        Settings.embed_model = self.Embedding_Model
        Settings.system_prompt = LLM_MetaData["Agent_Identity"]["content"]

    def Load_Construct_Index(self):
        if os.path.exists(os.path.join(LLM_MetaData["RAG_Database_Full_Directory"],"index_store.json")):
            self.Storage_Context = StorageContext.from_defaults(persist_dir=LLM_MetaData["RAG_Database_Directory"])
            self.Index = load_index_from_storage(self.Storage_Context)
            self.Query_Engine = self.Index.as_query_engine()
        else:
            self.Construct_Index()

    def CollectFiles(self,Directory):
        for File in os.listdir(Directory):
            FilePath = os.path.join(Directory,File)
            IsIncluded = not FilePath.endswith(Excluded_Context)
            if os.path.isfile(FilePath) and IsIncluded:
                try:
                    with open(FilePath,'r',encoding="utf-8") as F:
                        print(f"Added {File} to RAG database.\n")
                        yield (F.read(),FilePath)
                except Exception as E:
                    print(f"Excluded {File} from RAG database due to {E}")
                    pass
            elif os.path.isdir(FilePath) and IsIncluded:
                try:
                    for _ in self.CollectFiles(FilePath):
                        yield _
                except Exception as E:
                    print(f"Excluded {File} from RAG database due to {E}")
                    pass
            else:
                print(f"Excluded {File} explicitly from RAG database")

    def Construct_Index(self):
        Documents = []
        for Code,FilePath in self.CollectFiles(LLM_MetaData["RAG_Context_Full_Directory"]):
            Documents.append(Document(text=Code, metadata={"filepath": FilePath}))
        NodeProcessor = SentenceSplitter(chunk_size=512,chunk_overlap=50)
        Nodes = NodeProcessor.get_nodes_from_documents(Documents)
        self.Index = VectorStoreIndex(Nodes,embed_model=self.Embedding_Model)
        self.Query_Engine = self.Index.as_query_engine(similarity_top_k=5,response_mode="compact")
        self.Index.storage_context.persist(persist_dir=LLM_MetaData["RAG_Database_Directory"])

    def Fetch_Query(self,Query):
        print(f"Fetching query...")
        return self.Query_Engine.query(Query).response