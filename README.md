# Self-Correcting Text-to-SQL Agent

Ask a question in plain English, get an answer from your database. Unlike a chat model that writes SQL and stops, this agent executes what it writes, reads the failure, and repairs itself — looking up real column names when one is hallucinated, or sampling a column's actual values when a filter matches nothing.

Built with LangGraph and a custom MCP server that keeps the agent decoupled from the database behind read-only connections, query timeouts and row caps.
