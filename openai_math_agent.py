#!/usr/bin/env python3
"""Thin wrapper — delegates to agentic_graphs.examples.math_agent."""
import sys, runpy
sys.argv = ["agentic_graphs.examples.math_agent"] + sys.argv[1:]
runpy.run_module("agentic_graphs.examples.math_agent", run_name="__main__")
