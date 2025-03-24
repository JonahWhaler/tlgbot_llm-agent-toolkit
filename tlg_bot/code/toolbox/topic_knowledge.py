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
import myconfig

logger = logging.getLogger(__name__)


class TopicQueryTool(Tool):
    def __init__(self, vdb: chromadb.ClientAPI, title: str, encoder_config: dict):
        Tool.__init__(self, TopicQueryTool.function_info(title), False)
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
            logger.error("Failed to create %s knowledge base.", str(e))
            raise e

    def init(self, vdb, title, encoder: Encoder):
        for file_name in os.listdir(f"/assets/{title}"):
            file_path = f"/assets/{title}/{file_name}"
            logger.info("Loading file: %s", file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                K = max(len(text) // (encoder.ctx_length // 2), 1)
                if K == 1:
                    chunker = FixedGroupChunker(config={"K": 1})
                else:
                    chunker = SemanticChunker(
                        encoder=encoder,
                        config={
                            "K": K,
                            "MAX_ITERATION": 50,
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
    def function_info(title: str):
        description = f"""Query from {title} knowledge base.
        Use this tool to learn/find information related to {title}
        """
        return FunctionInfo(
            name=f"TopicQueryTool:{title}",
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
        if not self.validate(params=params):
            return json.dumps(
                {"error": "Invalid parameters for TopicQueryTool"},
                ensure_ascii=True,
            )

        params = json.loads(params)
        query = params.get("query", None)

        response = self.knowledge_base.query(
            query, return_n=5, output_types=["documents"]
        )["result"]["documents"]
        return json.dumps(response, ensure_ascii=False)

    async def run_async(self, params: str) -> str:
        return self.run(params)
