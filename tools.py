from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp import ClientSession
from mcp.types import (
    CallToolResult,
    TextContent,
)
from mcp.types import (
    Tool as MCPTool,
)

def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> str | list[str]:
    if call_tool_result.isError:
        raise ToolException(str(call_tool_result))

    text_contents = [
        content for content in call_tool_result.content 
        if isinstance(content, TextContent)
    ]

    if len(text_contents) == 1:
        return text_contents[0].text

    return [content.text for content in text_contents]

def _convert_mcp_to_langchain_tool(
    session: ClientSession,
    tool: MCPTool,
) -> BaseTool:
    async def call_tool(**arguments: dict[str, Any]) -> str | list[str]:
        call_tool_result = await session.call_tool(tool.name, arguments)
        return _convert_call_tool_result(call_tool_result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool,
    )

async def call_tool(tool_call: ToolCall, available_tools: list[BaseTool]) -> ToolMessage:
    tools_by_name = {tool.name: tool for tool in available_tools}
    tool = tools_by_name[tool_call["name"]]
    response = await tool.ainvoke(tool_call["args"])
    return ToolMessage(content=str(response), tool_call_id=tool_call["id"])

async def load_tools(session: ClientSession) -> list[BaseTool]:
    tools = await session.list_tools()
    return [_convert_mcp_to_langchain_tool(session, tool) for tool in tools.tools]
