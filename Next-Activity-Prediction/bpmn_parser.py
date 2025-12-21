import xml.etree.ElementTree as ET


class BPMNParser:
    def __init__(self, bpmn_path):
        self.tree = ET.parse(bpmn_path)
        self.root = self.tree.getroot()
        self.ns = self._detect_namespace()
        self.tasks = {}
        self._parse_tasks()

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

