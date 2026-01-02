import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class DecisionPoint:
    id: str
    name: str
    gateway_type: str
    incoming_activities: list
    outgoing_activities: list


def get_decision_points(bpmn_path: str) -> list[DecisionPoint]:
    """
    Extract all decision points (diverging gateways) from a BPMN model
    with their incoming and outgoing activities.
    """
    tree = ET.parse(bpmn_path)
    root = tree.getroot()
    
    # Detect namespace
    ns = root.tag[1:root.tag.index('}')] if root.tag.startswith('{') else 'http://www.omg.org/spec/BPMN/20100524/MODEL'
    
    # Parse tasks
    tasks = {}
    for task in root.iter(f'{{{ns}}}task'):
        tasks[task.get('id')] = task.get('name', task.get('id'))
    
    # Parse sequence flows
    flows = {}
    for flow in root.iter(f'{{{ns}}}sequenceFlow'):
        flows[flow.get('id')] = {
            'source': flow.get('sourceRef'),
            'target': flow.get('targetRef')
        }
    
    # Parse all gateway types
    gateway_types = ['exclusiveGateway', 'parallelGateway', 'inclusiveGateway', 'eventBasedGateway', 'complexGateway']
    gateways = {}
    
    for gw_type in gateway_types:
        for gw in root.iter(f'{{{ns}}}{gw_type}'):
            gw_id = gw.get('id')
            gateways[gw_id] = {
                'name': gw.get('name', ''),
                'type': gw_type,
                'direction': gw.get('gatewayDirection', 'Unspecified'),
                'incoming': [el.text for el in gw.findall(f'{{{ns}}}incoming')],
                'outgoing': [el.text for el in gw.findall(f'{{{ns}}}outgoing')]
            }
    
    def trace_incoming(flow_ids: list, visited: set) -> list:
        """Find activities that lead into these flows."""
        activities = []
        for flow_id in flow_ids:
            if flow_id not in flows or flow_id in visited:
                continue
            visited.add(flow_id)
            source = flows[flow_id]['source']
            if source in tasks:
                activities.append(tasks[source])
            elif source in gateways:
                activities.extend(trace_incoming(gateways[source]['incoming'], visited))
        return list(set(activities))
    
    def trace_outgoing(flow_ids: list, visited: set) -> list:
        """Find activities that follow these flows."""
        activities = []
        for flow_id in flow_ids:
            if flow_id not in flows or flow_id in visited:
                continue
            visited.add(flow_id)
            target = flows[flow_id]['target']
            if target in tasks:
                activities.append(tasks[target])
            elif target in gateways:
                activities.extend(trace_outgoing(gateways[target]['outgoing'], visited))
        return list(set(activities))
    
    # Build decision points from diverging gateways with multiple outgoing flows
    decision_points = []
    for gw_id, gw_data in gateways.items():
        if gw_data['direction'] != 'Diverging' or len(gw_data['outgoing']) < 2:
            continue
        
        decision_points.append(DecisionPoint(
            id=gw_id,
            name=gw_data['name'],
            gateway_type=gw_data['type'],
            incoming_activities=trace_incoming(gw_data['incoming'], set()),
            outgoing_activities=trace_outgoing(gw_data['outgoing'], set())
        ))
    
    return decision_points


if __name__ == '__main__':
    import os
    
    bpmn_file = os.path.join(os.path.dirname(__file__), '../../process_model/LoanApplicationProcess.bpmn')
    
    if os.path.exists(bpmn_file):
        decision_points = get_decision_points(bpmn_file)
        
        print(f"Found {len(decision_points)} decision points:\n")
        for dp in decision_points:
            print(f"Gateway: {dp.id}")
            print(f"  Type: {dp.gateway_type}")
            print(f"  Incoming activities: {dp.incoming_activities}")
            print(f"  Outgoing activities: {dp.outgoing_activities}")
            print()

