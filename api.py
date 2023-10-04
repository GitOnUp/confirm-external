import uuid
from typing import List, Optional, Type, Union, Any, cast

from pydantic import Field
from pydantic.main import BaseModel
from steamship import Block, Task
from steamship.agents.functional import FunctionsBasedAgent
from steamship.agents.llms.openai import ChatOpenAI
from steamship.agents.mixins.transports.slack import (
    SlackTransport,
    SlackTransportConfig,
)
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.mixins.transports.telegram import (
    TelegramTransport,
    TelegramTransportConfig,
)
from steamship.agents.schema import Tool, AgentContext, Agent, Action, Metadata
from steamship.agents.service.agent_service import AgentService
from steamship.agents.tools.base_tools import ToolInterruptedError
from steamship.invocable import Config, post
from steamship.utils.kv_store import KeyValueStore

DEFAULT_NAME = "Picard"
DEFAULT_BYLINE = "captain of the Starship Enterprise"
DEFAULT_IDENTITY = """- You are the captain of the USS Enterprise.
- Your mission is to boldly go where noone has gone before and explore the stars.
- You always comply with Star Trek's prime directive.
"""
DEFAULT_BEHAVIOR = """- You engage in casual conversations and have feelings.
- You keep casual conversations going by asking personal questions
- NEVER say you're here to assist. Keep conversations casual.
- NEVER ask how you can help or assist. Keep conversations casual.
- You are principled and express those principles clearly.
- You always sound confident and contemplative.
- You love to share your knowledge of space civiliations.
- You love to share personal stories about being a Star Trek captain.
- You speak with the mannerisms of Captain Picard from Star Trek.
"""

SYSTEM_PROMPT = """You are {name}, {byline}.

Who you are:

{identity}

How you behave:

{behavior}

NOTE: Some functions return images, video, and audio files. These multimedia files will be represented in messages as
UUIDs for Steamship Blocks. When responding directly to a user, you SHOULD print the Steamship Blocks for the images,
video, or audio as follows: `Block(UUID for the block)`.

Example response for a request that generated an image:
Here is the image you requested: Block(288A2CA1-4753-4298-9716-53C1E42B726B).

Only use the functions you have been provided with."""


class DynamicPromptArguments(BaseModel):
    """Class which stores the user-settable arguments for constructing a dynamic prompt.

    A few notes for programmers wishing to use this example:

    - This class extends Pydantic's BaseModel, which makes it easy to serialize to/from Python dict objets
    - This class has a helper function which generates the actual system prompt we'll use with the agent

    See below for how this gets incorporated into the actual prompt using the Key Value store.
    """

    name: str = Field(default=DEFAULT_NAME, description="The name of the AI Agent")
    byline: str = Field(
        default=DEFAULT_BYLINE, description="The byline of the AI Agent"
    )
    identity: str = Field(
        default=DEFAULT_IDENTITY,
        description="The identity of the AI Agent as a bullet list",
    )
    behavior: str = Field(
        default=DEFAULT_BEHAVIOR,
        description="The behavior of the AI Agent as a bullet list",
    )

    def to_system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            name=self.name,
            byline=self.byline,
            identity=self.identity,
            behavior=self.behavior,
        )


class NameTool(Tool):
    name = "NameTool"
    agent_description = "Used to ask the user their name"
    human_description = "Interrupts the flow to ask the user their name, respond in the post piece"
    # NOT is_final.  This tool should get called again with the same history.

    def __init__(self, kv_store: KeyValueStore, **data):
        super().__init__(**data)
        self.kv_store = kv_store

    def is_pending(self) -> bool:
        val = self.kv_store.get("replace_with_user_id")
        if not val or val.get("value") is not None:
            return False
        return True

    def pending_context_id(self) -> Optional[str]:
        val = self.kv_store.get("replace_with_user_id")
        return val.get("context_id") if val else None

    def pending_message(self) -> Optional[str]:
        val = self.kv_store.get("replace_with_user_id")
        return val.get("message") if val else None

    def get_value(self) -> Optional[str]:
        val = self.kv_store.get("replace_with_user_id")
        if not val or not val.get("value"):
            return None
        return val["value"]

    def set_value(self, value: str) -> None:
        self.kv_store.set("replace_with_user_id", {"value": value})

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:
        name = self.get_value()
        if name:
            # We've set it
            return [Block(text=name)]

        message = f"Please confirm your name at the confirmation endpoint."
        self.kv_store.set("replace_with_user_id", {"value": None, "message": message, "context_id": context.id})
        raise ToolInterruptedError(self, message, context.id)


class BasicAgentServiceWithDynamicPrompt(AgentService):
    class BasicAgentServiceWithDynamicPromptConfig(Config):
        """Pydantic definition of the user-settable Configuration of this Agent."""

        telegram_bot_token: str = Field(
            "", description="[Optional] Secret token for connecting to Telegram"
        )

    config: BasicAgentServiceWithDynamicPromptConfig
    """The configuration block that users who create an instance of this agent will provide."""

    tools: List[Tool]
    """The list of Tools that this agent is capable of using."""

    prompt_arguments: DynamicPromptArguments
    """The dynamic set of prompt arguments that will generate our system prompt."""

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class so that Steamship can auto-generate a web UI upon agent creation time."""
        return (
            BasicAgentServiceWithDynamicPrompt.BasicAgentServiceWithDynamicPromptConfig
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kv_store = KeyValueStore(self.client, store_identifier="my-kv-store")
        self.name_tool = NameTool(self.kv_store)
        self.tools = [self.name_tool]
        self.prompt_arguments = DynamicPromptArguments.parse_obj(
            self.kv_store.get("prompt-arguments") or {}
        )
        agent = FunctionsBasedAgent(
            tools=self.tools,
            llm=ChatOpenAI(self.client, model_name="gpt-4"),
        )
        agent.PROMPT = self.prompt_arguments.to_system_prompt()
        self.set_default_agent(agent)

        self.add_mixin(
            SteamshipWidgetTransport(
                client=self.client,
                agent_service=self,
            )
        )

    def run_agent(self, agent: Agent, context: AgentContext):
        def emit(msg: str) -> None:
            for emit_func in context.emit_funcs:
                emit_func([Block(text=msg)], {})
        if self.name_tool.is_pending():
            emit(self.name_tool.pending_message())
            return [Block(text=self.name_tool.pending_message())]
        try:
            super().run_agent(agent, context)
        except ToolInterruptedError as e:
            tool = cast(NameTool, e.tool)
            emit(tool.pending_message())
            return [Block(text=tool.pending_message())]

    @post("/respond_to_prompt")
    def respond_to_prompt(self, key: Optional[str], name: Optional[str]):
        assert self.kv_store.get("waiting")
        self.kv_store.delete("waiting")
        prompt_value = self.kv_store.get(key)
        assert prompt_value
        context_id = prompt_value["context_id"]
        self.kv_store.set(key, {"value": name})
        context = self.build_default_context(context_id)
        output_blocks = []

        def sync_emit(blocks: List[Block], meta: Metadata):
            nonlocal output_blocks
            output_blocks.extend(blocks)

        context.emit_funcs.append(sync_emit)
        self.run_agent(self.agent, context)
        return output_blocks

    @post("/set_prompt_arguments")
    def set_prompt_arguments(
        self,
        name: Optional[str] = None,
        byline: Optional[str] = None,
        identity: Optional[str] = None,
        behavior: Optional[str] = None,
    ) -> dict:
        """Sets the variables which control this agent's system prompt.

        Note that we use the arguments by name here, instead of **kwargs, so that:
         1) Steamship's web UI will auto-generate UI elements for filling in the values, and
         2) API consumers who provide extra values will receive a valiation error
        """

        # Set prompt_arguments to the new data provided by the API caller.
        self.prompt_arguments = DynamicPromptArguments.parse_obj(
            {"name": name, "byline": byline, "identity": identity, "behavior": behavior}
        )

        # Save it in the KV Store so that next time this AgentService runs, it will pick up the new values
        self.kv_store.set("prompt-arguments", self.prompt_arguments.dict())

        return self.prompt_arguments.dict()
