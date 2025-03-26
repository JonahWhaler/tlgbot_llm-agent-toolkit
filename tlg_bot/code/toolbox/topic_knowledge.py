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
    Encoder,
)
from llm_agent_toolkit.memory import ChromaMemory
from llm_agent_toolkit.chunkers import FixedGroupChunker, SemanticChunker
from llm_agent_toolkit.encoder import OllamaEncoder, OpenAIEncoder

logger = logging.getLogger(__name__)


class TopicQueryTool(Tool):
    """
    # Topic Query Tool

    Notes:
    - Please prepare all files in /assets/{title}
    - Only accept text files
    - Never swap encoder in between
    """

    def __init__(
        self,
        vdb: chromadb.ClientAPI,
        title: str,
        encoder_config: dict,
        num_results: int = 10,
    ):
        Tool.__init__(self, TopicQueryTool.function_info(title), False)
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
        self.num_results = num_results
        try:
            c = vdb.get_collection(name=title)
            logger.info("Count: %d", c.count())
            if c.count() == 0:
                logger.info("Creating new collection: %s", title)
                self.init(vdb, title, encoder)
                logger.info("Collection created: %s", title)
            # choice of chunker is not important
            # query does not need to be chunked
            chunker = FixedGroupChunker(
                config={"K": 1, "resolution": "back", "level": "character"}
            )
            chroma_memory = ChromaMemory(
                vdb=vdb, encoder=encoder, chunker=chunker, namespace=title
            )
            self.knowledge_base = chroma_memory
        except Exception as e:
            if f"Collection {title} does not exist." not in str(e):
                logger.error("TopicQueryTool: Exception: %s", e)
                raise

            logger.info("Creating new collection: %s", title)
            self.init(vdb, title, encoder)
            logger.info("Collection created: %s", title)
            chunker = FixedGroupChunker(
                config={"K": 1, "resolution": "back", "level": "character"}
            )
            chroma_memory = ChromaMemory(
                vdb=vdb, encoder=encoder, chunker=chunker, namespace=title
            )
            self.knowledge_base = chroma_memory

    def init(self, vdb, title, encoder: Encoder) -> None:
        """
        Initializes the knowledge base by processing all files in the specified directory.

        This method reads each file in the "/assets/{title}" directory, computes the
        appropriate chunk size `K`, and creates chunks of text using either a fixed
        or semantic chunking strategy. The chunks are then added to the ChromaMemory
        for the given namespace.

        Args:
            vdb: The vector database client to use for storing the knowledge base.
            title: The title of the knowledge base, which is used to locate files and
                as the namespace for storage.
            encoder (Encoder): The encoder used to process and encode the text content.
        """

        for file_name in os.listdir(f"/assets/{title}"):
            file_path = f"/assets/{title}/{file_name}"
            logger.info("Loading file: %s", file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                K = len(text) * 0.5 / min(encoder.ctx_length, 2048)
                K = max(int(K), 1)
                logger.info("K: %d", K)
                if K == 1:
                    chunker = FixedGroupChunker(config={"K": 1})
                else:
                    chunker = SemanticChunker(
                        encoder=encoder,
                        config={
                            "K": K,
                            "MAX_ITERATION": K * 10,
                            "update_rate": 0.3,
                            "min_coverage": 0.95,
                        },
                    )

                ChromaMemory(
                    vdb=vdb,
                    encoder=encoder,
                    chunker=chunker,
                    namespace=title,
                    overwrite=False,
                ).add(document_string=text)

    @staticmethod
    def function_info(title: str) -> FunctionInfo:
        """
        Generates a FunctionInfo object for the TopicQueryTool.

        This static method creates a FunctionInfo instance that describes
        the TopicQueryTool's capabilities for querying a specific knowledge
        base given by the title. It includes the tool's name, description,
        and the required parameters.

        Parameters:
            None

        Returns:
            FunctionInfo: An object containing the tool's metadata and expected
            input parameters.
        """

        description = f"""Query from {title} knowledge base.
        Use this tool to learn/find information related to {title}
        """
        return FunctionInfo(
            name=f"TopicQueryTool_{title}",
            description=description,
            parameters=FunctionParameters(
                properties=[
                    FunctionProperty(
                        name="query",
                        type=FunctionPropertyType.STRING,
                        description="Keyword that describe or define the query",
                    )
                ],
                type="object",
                required=["query"],
            ),
        )

    def run(self, params: str) -> str:
        """
        Run the TopicQueryTool with the given parameters.

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
            query, return_n=self.num_results, output_types=["documents"]
        )
        documents = response["result"]["documents"]
        return json.dumps(documents, ensure_ascii=False)

    async def run_async(self, params: str) -> str:
        """
        Run the TopicQueryTool with the given parameters asynchronously.

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
        return self.run(params)
