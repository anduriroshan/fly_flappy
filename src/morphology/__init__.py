"""Real traced neuron skeleton fetching for the web platform.

Fetches per-neuron SWC skeletons directly from Janelia's public MCNS GCS
bucket (CC-BY licensed, no login/token — same bucket Neuroglancer streams
from client-side). Consumed by server/brain.py to build the 3D morphology
the browser renders. Needs only `requests`, already a core dependency.
"""
from .gcs_fetch import fetch_skeletons_swc
