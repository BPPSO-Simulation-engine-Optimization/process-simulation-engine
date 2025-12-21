import xml.etree.ElementTree as ET


class BPMNParser:
    def __init__(self, bpmn_path):
        self.tree = ET.parse(bpmn_path)
        self.root = self.tree.getroot()
        self.ns = self._detect_namespace()
        self.tasks = {}
        self.gateways = {}
        self.flows = {}
        self._parse_tasks()
        self._parse_gateways()
        self._parse_flows()

    def _detect_namespace(self):
        tag = self.root.tag
        if tag.startswith('{'):
            return tag[1:tag.index('}')]
        return 'http://www.omg.org/spec/BPMN/20100524/MODEL'

    def _parse_tasks(self):
        for task in self.root.iter(f'{{{self.ns}}}task'):
            task_id = task.get('id')
            task_name = task.get('name', task_id)
            self.tasks[task_id] = task_name

    def _parse_gateways(self):
        for gw in self.root.iter(f'{{{self.ns}}}exclusiveGateway'):
            gw_id = gw.get('id')
            direction = gw.get('gatewayDirection', 'Unspecified')
            incoming = [e.text for e in gw.findall(f'{{{self.ns}}}incoming')]
            outgoing = [e.text for e in gw.findall(f'{{{self.ns}}}outgoing')]
            self.gateways[gw_id] = {
                'direction': direction,
                'incoming': incoming,
                'outgoing': outgoing
            }

    def _parse_flows(self):
        for flow in self.root.iter(f'{{{self.ns}}}sequenceFlow'):
            flow_id = flow.get('id')
            source = flow.get('sourceRef')
            target = flow.get('targetRef')
            self.flows[flow_id] = {'source': source, 'target': target}

    def get_xor_decision_points(self):
        decision_points = []
        for gw_id, gw_data in self.gateways.items():
            if gw_data['direction'] == 'Diverging' and len(gw_data['outgoing']) >= 2:
                decision_points.append(gw_id)
        return decision_points

