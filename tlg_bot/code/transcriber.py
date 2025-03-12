from llm_agent_toolkit.transcriber import AudioParameter, TranscriptionConfig
from llm_agent_toolkit.transcriber.whisper import LocalWhisperTranscriber
from llm_agent_toolkit.transcriber.open_ai import OpenAITranscriber


class TranscriberFactory:
    def __init__(
        self,
        provider: str,
        model_name: str,
        output_directory: str,
        audio_parameter: AudioParameter | None = None,
        transcript_config: TranscriptionConfig | None = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.output_directory = output_directory
        if audio_parameter is None:
            audio_parameter = AudioParameter()
        self.audio_parameter = audio_parameter
        if transcript_config is None:
            transcript_config = TranscriptionConfig(
                name=model_name, temperature=0.2, response_format="text"
            )
        self.transcript_config = transcript_config

    def get_transcriber(self):
        if self.provider == "openai":
            return OpenAITranscriber(self.transcript_config, self.audio_parameter)
        # local
        return LocalWhisperTranscriber(
            self.transcript_config, self.output_directory, self.audio_parameter
        )
