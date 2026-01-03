import re
from collections import defaultdict, OrderedDict


class DecisionPointExtractor:
    """Extracts decision point mappings from PM4Py BPMN models."""

    def __init__(self, bpmn_model):
        self._model = bpmn_model
        self._outgoing = defaultdict(list)
        self._incoming = defaultdict(list)
        self._parse_flows()

    def _parse_flows(self):
        for flow in self._model.get_flows():
            src, tgt = flow.source, flow.target
            src_name = getattr(src, "name", None) or getattr(src, "id", None)
            tgt_name = getattr(tgt, "name", None)

            if "Gateway" in src.__class__.__name__ and tgt_name:
                self._outgoing[src_name].append(tgt_name)

            if tgt_name and "Gateway" in tgt.__class__.__name__:
                self._incoming[tgt.name].append(src_name)

    def _sort_key(self, label):
        match = re.search(r"DP\s*(\d+)", label)
        return int(match.group(1)) if match else float("inf")

    def _resolve(self, mapping, key, visited=None):
        visited = set() if visited is None else visited
        result = []
        for name in mapping.get(key, []):
            name = name.strip()
            if name in mapping and name not in visited:
                visited.add(name)
                result.extend(self._resolve(mapping, name, visited.copy()))
            elif not name.startswith("DP") and not name.startswith("PG"):
                result.append(name)
        return result

    def _deduplicate(self, items):
        seen = set()
        return [x for x in items if x not in seen and not seen.add(x)]

    def extract(self):
        sorted_out = OrderedDict(sorted(self._outgoing.items(), key=lambda kv: self._sort_key(kv[0])))
        sorted_in = OrderedDict(sorted(self._incoming.items(), key=lambda kv: self._sort_key(kv[0])))

        resolved_out = {dp: self._deduplicate(self._resolve(sorted_out, dp)) for dp in sorted_out}
        resolved_in = {dp: self._deduplicate(self._resolve(sorted_in, dp)) for dp in sorted_in}

        all_dps = sorted(set(resolved_out) | set(resolved_in), key=self._sort_key)

        dp_map = {}
        for dp in all_dps:
            if dp.startswith("DP"):
                dp_map[dp] = {
                    "incoming": resolved_in.get(dp, []),
                    "outgoing": resolved_out.get(dp, []),
                }

        return dp_map

    @classmethod
    def from_model(cls, bpmn_model):
        return cls(bpmn_model).extract()


def extract_bpmn_decision_point_map(bpmn_model):
    """Legacy function wrapper for backward compatibility."""
    return DecisionPointExtractor.from_model(bpmn_model)

