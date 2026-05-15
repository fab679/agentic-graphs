#!/usr/bin/env python3
"""Thin wrapper — delegates to agentic_graphs.examples.debug_session."""
import sys, runpy
sys.argv = ["agentic_graphs.examples.debug_session"] + sys.argv[1:]
runpy.run_module("agentic_graphs.examples.debug_session", run_name="__main__")
