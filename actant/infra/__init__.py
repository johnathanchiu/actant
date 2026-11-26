"""Infrastructure the agent runtime runs on, usable without it: sandboxes, object storage,
assets and content blocks, and Temporal helpers.

Nothing here imports the agent layer (``actant.llm``, ``actant.tools``, ``actant.runtime``),
so this folder can become its own package by moving it.
"""
