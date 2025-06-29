import os

from typing import Dict, Optional, List
import logging

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    LoRAModulePath,
    PromptAdapterPath,
    OpenAIServingModels,
)

from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

logger = logging.getLogger("ray.serve")

app = FastAPI()

# Security scheme for bearer token
security = HTTPBearer()

# Load API tokens from environment variables
# You can set multiple tokens separated by commas
VALID_API_TOKENS = set(
    token.strip() 
    for token in os.environ.get('VLLM_API_TOKENS', '').split(',') 
    if token.strip()
)

# If no tokens are configured, use a default token (for development only)
if not VALID_API_TOKENS:
    DEFAULT_TOKEN = os.environ.get('VLLM_DEFAULT_API_TOKEN', 'your-secret-token-here')
    VALID_API_TOKENS.add(DEFAULT_TOKEN)
    logger.warning(f"No API tokens configured, using default token. Set VLLM_API_TOKENS environment variable for production use.")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify the provided API token"""
    token = credentials.credentials
    if token not in VALID_API_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request, token: str = Depends(verify_token)):
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            models = OpenAIServingModels(
                self.engine,
                model_config,
                [
                    BaseModelPath(
                        name=self.engine_args.model, model_path=self.engine_args.model
                    )
                ],
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
            )
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                models,
                self.response_role,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
                chat_template_content_format="auto",
                enable_reasoning=os.getenv("ENABLE_REASONING", 'False').lower() in ('true', '1', 't', 'yes'),
                reasoning_parser=os.getenv("REASONING_PARSER", None),
                enable_auto_tools=os.getenv("ENABLE_AUTO_TOOL_CHOICE", 'False').lower() in ('true', '1', 't', 'yes'),
                tool_parser=os.getenv("TOOL_CALL_PARSER", None),
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    @app.get("/health")
    async def health_check(self):
        """Health check endpoint - no authentication required"""
        return {"status": "healthy"}

    @app.get("/v1/models")
    async def list_models(self, token: str = Depends(verify_token)):
        """List available models - requires authentication"""
        model_name = os.environ.get('MODEL_ID', 'unknown').split('/')[-1]
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "vllm"
                }
            ]
        }

def parse_vllm_args(cli_args: Dict[str, str]):
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(arg_parser)

    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])

    logger.info(f"arg_strings: {arg_strings}")

    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


# serve run latest-serve:build_app model="Qwen/Qwen2.5-0.5B" tensor-parallel-size=1 accelerator="GPU"
def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    logger.info(f"parsed_args: {parsed_args}")

    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    engine_args.distributed_executor_backend = 'ray'

    logger.info(f"engine_args: {engine_args}")

    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
    )


model = build_app(
    {
        "model": os.environ['MODEL_ID'],
        "tensor-parallel-size": os.environ['TENSOR_PARALLELISM'],
        "pipeline-parallel-size": os.environ['PIPELINE_PARALLELISM'],
        "max-model-len": os.environ['MAX_MODEL_LEN'],
        "gpu-memory-utilization": os.environ['GPU_MEMORY_UTILIZATION'],
        "chat-template": os.getenv("CHAT_TEMPLATE", None),
     }
    )

logger.info(f"model: {model}")
