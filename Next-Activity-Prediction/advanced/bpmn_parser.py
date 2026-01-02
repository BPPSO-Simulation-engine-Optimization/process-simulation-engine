import re
import xml.etree.ElementTree as ET
from collections import defaultdict


class AdvancedBPMNParser:
    def __init__(self, bpmn_path):
        self.tree = ET.parse(bpmn_path)
        self.root = self.tree.getroot()
        self.ns = self._detect_namespace()
        self.gateways = {}
        self.named_nodes = {}
        self.flows = []
        self._parse_elements()
        self._parse_flows()

    def extract_decision_point_map(self):
        raw_outgoing = defaultdict(list)
        raw_incoming = defaultdict(list)

        for source_ref, target_ref in self.flows:
            source_is_gateway = source_ref in self.gateways
            target_is_gateway = target_ref in self.gateways
            source_name = self.gateways.get(source_ref) or self.named_nodes.get(source_ref)
            target_name = self.gateways.get(target_ref) or self.named_nodes.get(target_ref)

            if source_is_gateway and target_name:
                raw_outgoing[source_name].append(target_name)

            if target_is_gateway and source_name:
                raw_incoming[target_name].append(source_name)

        gateway_names = set(raw_outgoing.keys()) | set(raw_incoming.keys())
        ordered_gateways = sorted(gateway_names, key=self._gateway_sort_key)

        dp_map = {}
        for gateway_name in ordered_gateways:
            incoming = self._resolve_links(gateway_name, raw_incoming, set())
            outgoing = self._resolve_links(gateway_name, raw_outgoing, set())
            if gateway_name.startswith("DP"):
                dp_map[gateway_name] = {"incoming": incoming, "outgoing": outgoing}

        return dp_map

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

    def _detect_namespace(self):
        tag = self.root.tag
        if tag.startswith("{"):
            return tag[1 : tag.index("}")]
        return "http://www.omg.org/spec/BPMN/20100524/MODEL"

    def _parse_elements(self):
        gateway_tags = (
            "exclusiveGateway",
            "inclusiveGateway",
            "parallelGateway",
            "eventBasedGateway",
        )
        task_like_tags = (
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

        for tag in gateway_tags:
            for gateway in self.root.iter(f"{{{self.ns}}}{tag}"):
                gateway_id = gateway.get("id")
                gateway_name = gateway.get("name")
                if gateway_id and gateway_name:
                    self.gateways[gateway_id] = gateway_name
                    self.named_nodes[gateway_id] = gateway_name

        for tag in task_like_tags:
            for node in self.root.iter(f"{{{self.ns}}}{tag}"):
                node_id = node.get("id")
                node_name = node.get("name")
                if node_id and node_name:
                    self.named_nodes[node_id] = node_name

    def _parse_flows(self):
        for flow in self.root.iter(f"{{{self.ns}}}sequenceFlow"):
            source_ref = flow.get("sourceRef")
            target_ref = flow.get("targetRef")
            if source_ref and target_ref:
                self.flows.append((source_ref, target_ref))

    @staticmethod
    def _gateway_sort_key(name):
        match = re.search(r"\d+", name)
        return (int(match.group()) if match else float("inf"), name)

