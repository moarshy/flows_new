import ray
import yaml
from flows_new.components import (
    PrintComponent, 
    SquareComponent, 
    PDFReaderComponent, 
    OpenAIFAISSComponent, 
    LoadRetrieversComponent,
    ChunkerComponent,
    RetrieverComponent,
    QandAComponent
    )
from flows_new.blocks import Block
from flows_new.config import get_logger

logger = get_logger(__name__)


class WorkflowBuilder:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.blocks = self.build()

    def build(self):
        blocks = []
        for i, block_config in enumerate(self.config['blocks']):
            components = []
            for j, component_config in enumerate(block_config['components']):
                init_args = component_config.get('init_args', {})
                if component_config['type'] == 'PrintComponent':
                    component = PrintComponent(component_order=j, **init_args)
                    components.append(component)
                elif component_config['type'] == 'SquareComponent':
                    component = SquareComponent(component_order=j, **init_args)
                    components.append(component)
                elif component_config['type'] == 'PDFReaderComponent':
                    component = PDFReaderComponent(component_order=j, **init_args)
                    components.append(component)
                elif component_config['type'] == 'LoadRetrieversComponent':
                    component = LoadRetrieversComponent(component_order=j, **init_args)
                    components.append(component)
                elif component_config['type'] == 'OpenAIFAISSComponent':
                    component = OpenAIFAISSComponent(component_order=j, **init_args)
                    components.append(component)
                elif component_config['type'] == 'ChunkerComponent':
                    component = ChunkerComponent(component_order=j, **init_args)
                    components.append(component)
                elif component_config['type'] == 'RetrieverComponent':
                    component = RetrieverComponent(component_order=j, **init_args)
                    components.append(component)
                elif component_config['type'] == 'QandAComponent':
                    component = QandAComponent(component_order=j, **init_args)
                    components.append(component)
            block_type = block_config['block_type']
            block = Block(components, type=block_type)
            blocks.append(block)
        return blocks

    def execute(self):
        # Call ray.init() only if it hasn't been called before
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        for block in self.blocks:
            block.execute()
