#!/usr/bin/env python3
"""Thin wrapper — delegates to agentic_graphs.examples.multi_agent_demo."""
import sys, runpy
sys.argv = ["agentic_graphs.examples.multi_agent_demo"] + sys.argv[1:]
runpy.run_module("agentic_graphs.examples.multi_agent_demo", run_name="__main__")
