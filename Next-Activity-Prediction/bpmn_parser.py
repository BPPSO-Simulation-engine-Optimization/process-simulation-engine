import xml.etree.ElementTree as ET


class BPMNParser:
    def __init__(self, bpmn_path):
        self.tree = ET.parse(bpmn_path)
        self.root = self.tree.getroot()
        self.ns = self._detect_namespace()

    def _detect_namespace(self):
        tag = self.root.tag
        if tag.startswith('{'):
            return tag[1:tag.index('}')]
        return 'http://www.omg.org/spec/BPMN/20100524/MODEL'

