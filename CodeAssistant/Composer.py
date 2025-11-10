from . import \
    ASR , \
    llama_Index_VDB, \
    Supported_Files, Supported_Files_Keys

class Agent_Composer():
    def Boot(self):
        self.ASR_Agent = ASR.ASR_Pipeline()
        self.RAG_Agent = llama_Index_VDB.RAG_VDB()
        self.RAG_Agent.Load_Construct_Index()

    def RunAgent(self):
        print("type 't' for text prompts, 'v' for voice prompts, or 'exit' to quit.")
        while True:
            Option = str(input(">> ")).lower()
            if Option == "t":
                Prompt = str(input("Type text prompt >> "))
                Response = self.RAG_Agent.Fetch_Query(Prompt).split("```")
                self.Generate_CodeFiles(Response)

            elif Option == "v":
                Detected_Audio = self.ASR_Agent.RunInference()["text"]
                print(f"\n\n------------------------------\nAudio detected:{Detected_Audio}\n------------------------------\n")
                print("Run prompt? (y/n/exit) ")
                while True:
                    Accept_Prompt = str(input(">> ")).lower()
                    if Accept_Prompt == 'y':
                        Response = self.RAG_Agent.Fetch_Query(Detected_Audio).split("```")
                        self.Generate_CodeFiles(Response)
                        break

                    elif Accept_Prompt == 'n':
                        Detected_Audio = self.ASR_Agent.RunInference()["text"]
                        print(f"\n\n------------------------------\nAudio detected:{Detected_Audio}\n------------------------------\n")
                        print("Run prompt? (y/n/exit) ")

                    elif Accept_Prompt == 'exit':
                        break

                    else:
                        print(f"{Accept_Prompt} is not valid, type 'y' for yes, 'n' for no, or 'exit' to return to the main menu.")

            elif Option == "exit":
                break

            else:
                print(f"{Option} is not valid, type 't' for text prompts, 'v' for voice prompts, or 'exit' to quit.")
        exit()

    def Generate_CodeFiles(self,Response):
        for Chunk in Response:
            print(Chunk)
            Programming_Language = Chunk.split("\n")[0].replace('```','')
            print(Programming_Language)
            if(Programming_Language in Supported_Files_Keys):
                with open(f"./Isolated_Files/Isolated_Script{Supported_Files[Programming_Language]}", "w+", encoding='utf-8') as CodeFile:
                    CodeFile.write(Chunk.replace('```','').replace(Programming_Language,'')) 