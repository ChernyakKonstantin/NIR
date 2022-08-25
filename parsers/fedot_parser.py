"""
The module contains a parser for FEDOT pipelines.
The parser parse JSON files.
"""
import json
import numpy as np
import networkx as nx
from typing import Any

class FedotParser:
    def __init__(self):
        pass


    def json2nx(self, pipeline_path: str) -> "networkx.DiGraph":
        """The method parse saved pipeline json-file. `networx.DiGraph` object is returned.
        :param pipeline_path: path to pipeline json-file.
        """
        with open(pipeline_path, "r") as f:
            pipeline_desc = json.load(f)
        nodes = pipeline_desc["nodes"]
        graph = nx.DiGraph()
        for node in nodes:
            op_id = node["operation_id"]
            graph.add_node(
                op_id,
                operation_type = node["operation_type"],
                # params = node["params"], # TODO: Somehow check what it can be and what are defaults
                # custom_params = node["custom_params"], # TODO: Somehow check what it can be and what are defaults
            )
            for prev_op_id in node["nodes_from"]:
                graph.add_edge(prev_op_id, op_id)
        return graph