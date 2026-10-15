"""Small package marker so ``exercises.agent`` / ``exercises.cli`` import cleanly.

The exercise scripts themselves are run as files (``python exercises/exercise_1...``)
but the shared graph lives in ``exercises/agent.py`` and is imported by ``app.py``
and the test suite, which needs the directory to be a real package.
"""
