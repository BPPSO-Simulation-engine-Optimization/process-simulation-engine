import re
import xml.etree.ElementTree as ET
from collections import defaultdict


class BPMNParser:
    """Parses BPMN XML files and extracts decision point mappings."""

    GATEWAY_TAGS = (
        "exclusiveGateway",
        "inclusiveGateway",
        "parallelGateway",
        "eventBasedGateway",
    )

    TASK_TAGS = (
        "task",
        "userTask",
        "serviceTask",
        "scriptTask",
        "businessRuleTask",
        "manualTask",
        "sendTask",
        "receiveTask",
        "callActivity",
        "subProcess",
        "startEvent",
        "endEvent",
        "intermediateThrowEvent",
        "intermediateCatchEvent",
    )

    def __init__(self, bpmn_path):
        self._tree = ET.parse(bpmn_path)
        self._root = self._tree.getroot()
        self._ns = self._detect_namespace()
        self._gateways = {}
        self._named_nodes = {}
        self._flows = []
        self._parse()

    def _detect_namespace(self):
        tag = self._root.tag
        if tag.startswith("{"):
            return tag[1:tag.index("}")]
        return "http://www.omg.org/spec/BPMN/20100524/MODEL"

    def _parse(self):
        self._parse_gateways()
        self._parse_tasks()
        self._parse_flows()

    def _parse_gateways(self):
        for tag in self.GATEWAY_TAGS:
            for elem in self._root.iter(f"{{{self._ns}}}{tag}"):
                elem_id = elem.get("id")
                elem_name = elem.get("name")
                if elem_id and elem_name:
                    self._gateways[elem_id] = elem_name
                    self._named_nodes[elem_id] = elem_name

    def _parse_tasks(self):
        for tag in self.TASK_TAGS:
            for elem in self._root.iter(f"{{{self._ns}}}{tag}"):
                elem_id = elem.get("id")
                elem_name = elem.get("name")
                if elem_id and elem_name:
                    self._named_nodes[elem_id] = elem_name

    def _parse_flows(self):
        for flow in self._root.iter(f"{{{self._ns}}}sequenceFlow"):
            src = flow.get("sourceRef")
            tgt = flow.get("targetRef")
            if src and tgt:
                self._flows.append((src, tgt))

    @staticmethod
    def _sort_key(name):
        match = re.search(r"\d+", name)
        return (int(match.group()) if match else float("inf"), name)

    def _resolve_links(self, gateway_name, raw_map, visiting):
        if gateway_name in visiting:
            return []

        visiting = visiting | {gateway_name}
        resolved = []

        for name in raw_map.get(gateway_name, []):
            if name.startswith(("DP", "PG")):
                nested = self._resolve_links(name, raw_map, visiting)
                for item in nested:
                    if item not in resolved:
                        resolved.append(item)
            else:
                if name not in resolved:
                    resolved.append(name)

        return resolved

    def extract_decision_point_map(self):
        raw_out = defaultdict(list)
        raw_in = defaultdict(list)

        for src_ref, tgt_ref in self._flows:
            src_is_gw = src_ref in self._gateways
            tgt_is_gw = tgt_ref in self._gateways
            src_name = self._gateways.get(src_ref) or self._named_nodes.get(src_ref)
            tgt_name = self._gateways.get(tgt_ref) or self._named_nodes.get(tgt_ref)

            if src_is_gw and tgt_name:
                raw_out[src_name].append(tgt_name)
            if tgt_is_gw and src_name:
                raw_in[tgt_name].append(src_name)

        gw_names = set(raw_out.keys()) | set(raw_in.keys())
        ordered = sorted(gw_names, key=self._sort_key)

        dp_map = {}
        for gw in ordered:
            incoming = self._resolve_links(gw, raw_in, set())
            outgoing = self._resolve_links(gw, raw_out, set())
            if gw.startswith("DP"):
                dp_map[gw] = {"incoming": incoming, "outgoing": outgoing}

        return dp_map


# Alias for backward compatibility
AdvancedBPMNParser = BPMNParser

