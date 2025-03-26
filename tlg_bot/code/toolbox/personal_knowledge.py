"""
Module DocString
"""

import os
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

logger = logging.getLogger(__name__)


class PersonalKnowledgeBaseTool(Tool):
    """
    # Personal Knowledge Base Tool

    Notes:
    - Please do not move the user_vdb
    - Never swap encoder in between
    - This tool only read from the user_vdb,
    - Other parts of the code will add and remove content from the user_vdb
    """

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
                connection_string=os.environ["OLLAMA_HOST"],
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
        """
        Generates a FunctionInfo object for the PersonalKnowledgeBaseTool.

        This static method creates a FunctionInfo instance that describes
        the PersonalKnowledgeBaseTool's capabilities for querying a specific knowledge
        base given by the title. It includes the tool's name, description,
        and the required parameters.

        Parameters:
            None

        Returns:
            A FunctionInfo object containing the tool's metadata and expected
            input parameters.
        """
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
        """
        Asynchronously run the PersonalKnowledgeBaseTool with the given parameters.

        This method validates the provided parameters, extracts the query keyword,
        and queries the personal knowledge base to retrieve relevant documents.
        It returns the result in JSON format.

        Parameters:
            params (str): JSON string containing the query keyword.

        Returns:
            str: JSON string containing the retrieved documents or an error message
            if the parameters are invalid.
        """
        params_dict = json.loads(params)
        valid, validation_message = self.validate(**params_dict)
        if not valid:
            return json.dumps(
                {
                    "error": "Invalid Parameters",
                    "detail": validation_message,
                },
                ensure_ascii=False,
            )
        # Load parameters
        query = params_dict.get("query", None)
        response = self.knowledge_base.query(
            query_string=query, return_n=self.num_results, output_types=["documents"]
        )
        documents = response["result"]["documents"]
        return json.dumps(documents, ensure_ascii=False)

    def run(self, params: str) -> str:
        """
        Run the PersonalKnowledgeBaseTool with the given parameters.

        Parameters:
            params (str): JSON string containing the parameters of the tool.

        Returns:
            str: JSON string containing the retrieved documents or an error message
            if the parameters are invalid.
        """
        params_dict = json.loads(params)
        valid, validation_message = self.validate(**params_dict)
        if not valid:
            return json.dumps(
                {
                    "error": "Invalid Parameters",
                    "detail": validation_message,
                },
                ensure_ascii=False,
            )
        # Load parameters
        query = params_dict.get("query", None)
        response = self.knowledge_base.query(
            query_string=query, return_n=self.num_results, output_types=["documents"]
        )
        documents = response["result"]["documents"]
        return json.dumps(documents, ensure_ascii=False)
