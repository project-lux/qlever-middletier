"""Domain logic: one service per thing the API can do.

Each takes its collaborators (settings, catalogue, clients) as constructor
arguments, so any of them can be built and exercised on its own.
"""
