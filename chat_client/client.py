import contextlib
from mcp import ClientSession
from mcp.client.sse import sse_client
from common.config import Config

def server_url():
    return f"http://{Config.Server.HOST}:{Config.Server.PORT}{Config.Server.SSE_PATH}"

@contextlib.asynccontextmanager
async def connect_to_server(url: str = server_url()):
    async with sse_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session
