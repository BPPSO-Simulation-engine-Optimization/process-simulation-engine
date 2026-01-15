"""
Extracts XOR gateways and their connections from BPMN models.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class XORGateway:
    id: str
    name: str
    incoming_flows: list
    outgoing_flows: list
    preceding_activities: list
    succeeding_activities: list


class GatewayExtractor:
    BPMN_NS = {
        'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'
    }

    def __init__(self, bpmn_path: str):
        self.bpmn_path = bpmn_path
        self.tree = ET.parse(bpmn_path)
        self.root = self.tree.getroot()
        self._detect_namespace()
        
        self.tasks = {}
        self.gateways = {}
        self.flows = {}
        
        self._parse_elements()

    def _detect_namespace(self):
        """Detect the default namespace from the root element."""
        tag = self.root.tag
        if tag.startswith('{'):
            self.default_ns = tag[1:tag.index('}')]
            self.BPMN_NS['bpmn'] = self.default_ns
        else:
            self.default_ns = self.BPMN_NS['bpmn']

    def _find_all(self, tag: str):
        """Find all elements with given tag, handling namespace."""
        return self.root.iter(f'{{{self.default_ns}}}{tag}')

    def _parse_elements(self):
        """Parse all relevant BPMN elements."""
        # Parse tasks
        for task in self._find_all('task'):
            task_id = task.get('id')
            task_name = task.get('name', task_id)
            self.tasks[task_id] = task_name

        # Parse exclusive gateways (XOR)
        for gateway in self._find_all('exclusiveGateway'):
            gw_id = gateway.get('id')
            gw_name = gateway.get('name', '')
            direction = gateway.get('gatewayDirection', 'Unspecified')
            
            incoming = [el.text for el in gateway.findall(f'{{{self.default_ns}}}incoming')]
            outgoing = [el.text for el in gateway.findall(f'{{{self.default_ns}}}outgoing')]
            
            self.gateways[gw_id] = {
                'name': gw_name,
                'direction': direction,
                'incoming': incoming,
                'outgoing': outgoing
            }

        # Parse sequence flows
        for flow in self._find_all('sequenceFlow'):
            flow_id = flow.get('id')
            source = flow.get('sourceRef')
            target = flow.get('targetRef')
            self.flows[flow_id] = {'source': source, 'target': target}

    def get_diverging_xor_gateways(self) -> list[XORGateway]:
        """Get all diverging XOR gateways (decision points)."""
        xor_gateways = []
        
        for gw_id, gw_data in self.gateways.items():
            # Only diverging gateways with multiple outgoing flows are decision points
            if gw_data['direction'] != 'Diverging' or len(gw_data['outgoing']) < 2:
                continue
            
            preceding = self._get_preceding_activities(gw_id, gw_data['incoming'], set())
            succeeding = self._get_succeeding_activities(gw_data['outgoing'], set())
            
            xor_gateways.append(XORGateway(
                id=gw_id,
                name=gw_data['name'],
                incoming_flows=gw_data['incoming'],
                outgoing_flows=gw_data['outgoing'],
                preceding_activities=preceding,
                succeeding_activities=succeeding
            ))
        
        return xor_gateways

    def _get_preceding_activities(self, gateway_id: str, incoming_flows: list, visited: set) -> list:
        """Find activities that directly precede this gateway."""
        preceding = []
        for flow_id in incoming_flows:
            if flow_id not in self.flows or flow_id in visited:
                continue
            visited.add(flow_id)
            
            source_id = self.flows[flow_id]['source']
            if source_id in self.tasks:
                preceding.append(self.tasks[source_id])
            elif source_id in self.gateways:
                inner_incoming = self.gateways[source_id]['incoming']
                preceding.extend(self._get_preceding_activities(source_id, inner_incoming, visited))
        return list(set(preceding))

    def _get_succeeding_activities(self, outgoing_flows: list, visited: set) -> list:
        """Find activities that directly follow this gateway's branches."""
        succeeding = []
        for flow_id in outgoing_flows:
            if flow_id not in self.flows or flow_id in visited:
                continue
            visited.add(flow_id)
            
            target_id = self.flows[flow_id]['target']
            if target_id in self.tasks:
                succeeding.append(self.tasks[target_id])
            elif target_id in self.gateways:
                # Gateway after gateway - trace forward
                inner_outgoing = self.gateways[target_id]['outgoing']
                succeeding.extend(self._get_succeeding_activities(inner_outgoing, visited))
        return list(set(succeeding))

    def get_activity_to_gateway_map(self) -> dict:
        """Create mapping: activity -> list of gateways it can lead to."""
        activity_gateway_map = {}
        
        for gateway in self.get_diverging_xor_gateways():
            for activity in gateway.preceding_activities:
                if activity not in activity_gateway_map:
                    activity_gateway_map[activity] = []
                activity_gateway_map[activity].append({
                    'gateway_id': gateway.id,
                    'possible_branches': gateway.succeeding_activities
                })
        
        return activity_gateway_map

    def get_gateway_branches(self) -> dict:
        """Get mapping: gateway_id -> list of possible next activities."""
        return {
            gw.id: gw.succeeding_activities 
            for gw in self.get_diverging_xor_gateways()
        }

