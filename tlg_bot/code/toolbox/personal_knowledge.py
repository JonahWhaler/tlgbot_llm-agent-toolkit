import logging
import json
import chromadb
from llm_agent_toolkit import (
    Tool,
    FunctionInfo,
    FunctionProperty,
    FunctionPropertyType,
    FunctionParameters,
)
from llm_agent_toolkit.memory import ChromaMemory
from llm_agent_toolkit.chunkers import FixedGroupChunker
from llm_agent_toolkit.encoder import OllamaEncoder, OpenAIEncoder
import myconfig

logger = logging.getLogger(__name__)


class PersonalKnowledgeBaseTool(Tool):
    def __init__(
        self, user_vdb: chromadb.ClientAPI, encoder_config: dict, num_results: int = 10
    ):
        Tool.__init__(self, PersonalKnowledgeBaseTool.function_info(), True)
        self.user_vdb = user_vdb
        if encoder_config["provider"] == "openai":
            encoder = OpenAIEncoder(
                model_name=encoder_config["model_name"],
                dimension=encoder_config["dimension"],
            )
        else:
            encoder = OllamaEncoder(
                connection_string=myconfig.OLLAMA_HOST,
                model_name=encoder_config["model_name"],
            )
        chunker = FixedGroupChunker(
            config={"K": 1, "resolution": "back", "level": "character"}
        )
        chroma_memory = ChromaMemory(vdb=user_vdb, encoder=encoder, chunker=chunker)
        self.knowledge_base = chroma_memory
        self.num_results = num_results

    @staticmethod
    def function_info():
        description = """
        Use this tool to search the personal knowledge base.
        All files uploaded by the user are stored here.

        Pass in the query keyword and this tool will return you the relevant chunks.
        """
        return FunctionInfo(
            name="PersonalKnowledgeBaseTool",
            description=description,
            parameters=FunctionParameters(
                properties=[
                    FunctionProperty(
                        name="query",
                        type=FunctionPropertyType.STRING,
                        description="The query keyword. Please avoid vague and short keyword.",
                    ),
                ],
                type="object",
                required=["query"],
            ),
        )

    async def run_async(self, params: str) -> str:
        if not self.validate(params=params):
            return {"error": "Invalid parameters for PersonalKnowledgeBaseTool"}

        # Load parameters
        params = json.loads(params)

        query = params.get("query", None)

        response = self.knowledge_base.query(
            query_string=query, return_n=self.num_results, output_types=["documents"]
        )
        return json.dumps(response, ensure_ascii=False)

    def run(self, params: str) -> str:
        if not self.validate(params=params):
            return {"error": "Invalid parameters for PersonalKnowledgeBaseTool"}

        # Load parameters
        params = json.loads(params)

        query = params.get("query", None)

        response = self.knowledge_base.query(
            query_string=query, return_n=self.num_results, output_types=["documents"]
        )
        return json.dumps(response, ensure_ascii=False)
