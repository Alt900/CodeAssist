from . import \
    torch, np, \
    evaluate, \
    Audio_MetaData, \
    AutoProcessor, pipeline, \
    pyaudio

class ASR_Pipeline():
    def __init__(self,UseGPU=True):
        self.processor = AutoProcessor.from_pretrained(Audio_MetaData["ASR_Model"])
        self.wer = evaluate.load("wer")
        self.Device = 0 if (torch.cuda.is_available() if UseGPU else False) else -1

    def RunInference(self):

        CurrentInputFrames = []
        PyAudioObj = pyaudio.PyAudio()

        MicStream = PyAudioObj.open(
            format=Audio_MetaData["Format"],
            channels=Audio_MetaData["Channels"],
            rate=Audio_MetaData["Sample_Rate"],
            input=True,
            frames_per_buffer=Audio_MetaData["Chunks"]
        )
        
        print("\n\n------------------------------")
        print("Audio is recording...")
        print("------------------------------\n")

        for _ in range(0,int(Audio_MetaData["Sample_Rate"]/Audio_MetaData["Chunks"]+Audio_MetaData["Clip_Time_Length"])):
            data = MicStream.read(Audio_MetaData["Chunks"])
            CurrentInputFrames.append(np.frombuffer(data,dtype=np.int16))
        
        MicStream.stop_stream()
        MicStream.close()
        PyAudioObj.terminate()

        Audio_Array = np.concatenate(CurrentInputFrames).astype(np.float32) / 32768.0

        transcriber = pipeline(
            "automatic-speech-recognition", 
            model = Audio_MetaData["ASR_Model"],
            device = self.Device
        )
        return transcriber(
            {"raw":Audio_Array,"sampling_rate":Audio_MetaData["Sample_Rate"]}
        )